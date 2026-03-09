#!/usr/bin/env python3
"""Run self-critique representation drift experiments with activation analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Some sandbox users have no passwd entry; torch/getpass needs USER fallback.
os.environ.setdefault("USER", "codex")
os.environ.setdefault("LOGNAME", "codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_from_disk
from openai import OpenAI
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def parse_last_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def extract_choice_letter(text: str) -> str | None:
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else None


def score_gsm8k(pred: str, gold: str) -> int:
    p = parse_last_number(pred)
    g = parse_last_number(gold)
    if p is None or g is None:
        return 0
    try:
        return int(abs(float(p) - float(g)) < 1e-6)
    except ValueError:
        return 0


def score_csqa(pred: str, gold_letter: str) -> int:
    return int(extract_choice_letter(pred or "") == gold_letter)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots.append(np.mean(sample))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


@dataclass
class Example:
    example_id: str
    dataset: str
    prompt: str
    gold: str
    meta: dict[str, Any]


class LocalLLMRunner:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def layer_vectors(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        # hidden_states includes embedding output at index 0; keep only transformer layers.
        hs = outputs.hidden_states[1:]
        vecs = torch.stack([layer[0, -1, :].float().cpu() for layer in hs], dim=0)
        return vecs.numpy()


class ExternalCritiqueClient:
    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def critique(self, question: str, answer: str, timeout_s: float = 60.0) -> str:
        prompt = (
            "You are a strict reasoning critic. Identify concrete flaws in the answer and "
            "provide concise guidance to fix them. Return bullet points only.\n\n"
            f"Question:\n{question}\n\n"
            f"Proposed answer:\n{answer}\n"
        )
        start = time.time()
        for attempt in range(3):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=0.0,
                    max_output_tokens=200,
                )
                return (resp.output_text or "").strip()
            except Exception as exc:  # noqa: BLE001
                if time.time() - start > timeout_s or attempt == 2:
                    return f"External critique unavailable: {exc}"
                time.sleep(1.5 * (attempt + 1))
        return "External critique unavailable."


def build_examples(gsm8k_path: Path, csqa_path: Path, n_gsm: int, n_csqa: int, seed: int) -> list[Example]:
    rng = np.random.default_rng(seed)
    examples: list[Example] = []

    gsm = load_from_disk(str(gsm8k_path))
    gsm_test = gsm["test"]
    gsm_idxs = rng.choice(len(gsm_test), size=min(n_gsm, len(gsm_test)), replace=False)
    for idx in gsm_idxs:
        item = gsm_test[int(idx)]
        q = item["question"]
        examples.append(
            Example(
                example_id=f"gsm8k_{idx}",
                dataset="gsm8k",
                prompt=(
                    "Solve the following math problem. Show concise reasoning, then end with "
                    "'Final Answer: <number>'.\n\n"
                    f"Problem: {q}\n"
                ),
                gold=item["answer"],
                meta={"question": q},
            )
        )

    csqa = load_from_disk(str(csqa_path))
    csqa_val = csqa["validation"]
    csqa_idxs = rng.choice(len(csqa_val), size=min(n_csqa, len(csqa_val)), replace=False)
    for idx in csqa_idxs:
        item = csqa_val[int(idx)]
        choices = item["choices"]
        option_lines = [f"{lab}. {txt}" for lab, txt in zip(choices["label"], choices["text"], strict=True)]
        q = item["question"]
        examples.append(
            Example(
                example_id=f"csqa_{idx}",
                dataset="commonsense_qa",
                prompt=(
                    "Answer the multiple-choice question. Explain briefly, then end with "
                    "'Final Answer: <A/B/C/D/E>'.\n\n"
                    f"Question: {q}\nOptions:\n" + "\n".join(option_lines) + "\n"
                ),
                gold=item["answerKey"],
                meta={"question": q, "options": option_lines},
            )
        )

    random.shuffle(examples)
    return examples


def score_example(dataset: str, prediction: str, gold: str) -> int:
    if dataset == "gsm8k":
        return score_gsm8k(prediction, gold)
    return score_csqa(prediction, gold)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - np.dot(a, b) / denom)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    root = Path(args.workspace_root).resolve()
    results_dir = root / "results"
    plots_dir = results_dir / "plots"
    model_out_dir = results_dir / "model_outputs"
    eval_dir = results_dir / "evaluations"
    for d in [results_dir, plots_dir, model_out_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    examples = build_examples(
        root / "datasets" / "gsm8k",
        root / "datasets" / "commonsense_qa",
        args.n_gsm8k,
        args.n_csqa,
        args.seed,
    )

    local_llm = LocalLLMRunner(args.local_model)
    ext_client = ExternalCritiqueClient(args.external_model) if (args.use_external and os.getenv("OPENAI_API_KEY")) else None

    records: list[dict[str, Any]] = []
    layer_cos_rows: list[dict[str, Any]] = []
    layer_l2_rows: list[dict[str, Any]] = []
    ext_calls = 0

    for i, ex in enumerate(examples):
        baseline = local_llm.generate(ex.prompt, max_new_tokens=args.max_new_tokens, temperature=0.0)
        red_prompt = (
            ex.prompt
            + "\nYour previous answer may be wrong. Re-answer from scratch and improve it. "
            "End with Final Answer format.\n"
        )
        redecode = local_llm.generate(red_prompt, max_new_tokens=args.max_new_tokens, temperature=0.0)

        critique_prompt = (
            ex.prompt
            + f"\nInitial answer:\n{baseline}\n\n"
            "Critique this answer. Find concrete reasoning errors or missing checks. "
            "Be specific and concise."
        )
        self_critique = local_llm.generate(critique_prompt, max_new_tokens=128, temperature=0.0)

        revise_prompt = (
            ex.prompt
            + f"\nInitial answer:\n{baseline}\n\nSelf-critique:\n{self_critique}\n\n"
            "Now provide a revised answer that fixes the issues. End with Final Answer format."
        )
        self_revised = local_llm.generate(revise_prompt, max_new_tokens=args.max_new_tokens, temperature=0.0)

        external_critique = ""
        external_revised = ""
        if ext_client is not None and ext_calls < args.n_external:
            external_critique = ext_client.critique(ex.meta["question"], baseline)
            ext_revise_prompt = (
                ex.prompt
                + f"\nInitial answer:\n{baseline}\n\nExternal critique:\n{external_critique}\n\n"
                "Now provide a revised answer that addresses the critique. End with Final Answer format."
            )
            external_revised = local_llm.generate(ext_revise_prompt, max_new_tokens=args.max_new_tokens, temperature=0.0)
            ext_calls += 1

        b_score = score_example(ex.dataset, baseline, ex.gold)
        r_score = score_example(ex.dataset, redecode, ex.gold)
        s_score = score_example(ex.dataset, self_revised, ex.gold)
        e_score = score_example(ex.dataset, external_revised, ex.gold) if external_revised else np.nan

        b_vec = local_llm.layer_vectors(ex.prompt + "\n" + baseline)
        r_vec = local_llm.layer_vectors(ex.prompt + "\n" + redecode)
        s_vec = local_llm.layer_vectors(ex.prompt + "\n" + self_revised)
        e_vec = local_llm.layer_vectors(ex.prompt + "\n" + external_revised) if external_revised else None

        n_layers = b_vec.shape[0]
        for layer_idx in range(n_layers):
            base = b_vec[layer_idx]
            red = r_vec[layer_idx]
            selfv = s_vec[layer_idx]
            layer_cos_rows.append(
                {
                    "example_id": ex.example_id,
                    "dataset": ex.dataset,
                    "layer": layer_idx + 1,
                    "condition": "redecode",
                    "cosine_distance": cosine_distance(base, red),
                }
            )
            layer_cos_rows.append(
                {
                    "example_id": ex.example_id,
                    "dataset": ex.dataset,
                    "layer": layer_idx + 1,
                    "condition": "self_critique",
                    "cosine_distance": cosine_distance(base, selfv),
                }
            )
            layer_l2_rows.append(
                {
                    "example_id": ex.example_id,
                    "dataset": ex.dataset,
                    "layer": layer_idx + 1,
                    "condition": "redecode",
                    "l2": float(np.linalg.norm(red - base)),
                }
            )
            layer_l2_rows.append(
                {
                    "example_id": ex.example_id,
                    "dataset": ex.dataset,
                    "layer": layer_idx + 1,
                    "condition": "self_critique",
                    "l2": float(np.linalg.norm(selfv - base)),
                }
            )
            if e_vec is not None:
                extv = e_vec[layer_idx]
                layer_cos_rows.append(
                    {
                        "example_id": ex.example_id,
                        "dataset": ex.dataset,
                        "layer": layer_idx + 1,
                        "condition": "external_critique",
                        "cosine_distance": cosine_distance(base, extv),
                    }
                )
                layer_l2_rows.append(
                    {
                        "example_id": ex.example_id,
                        "dataset": ex.dataset,
                        "layer": layer_idx + 1,
                        "condition": "external_critique",
                        "l2": float(np.linalg.norm(extv - base)),
                    }
                )

        drift_vec = (s_vec - b_vec).reshape(-1)
        drift_norm = float(np.linalg.norm(drift_vec))
        records.append(
            {
                "example_id": ex.example_id,
                "dataset": ex.dataset,
                "question": ex.meta["question"],
                "gold": ex.gold,
                "baseline_answer": baseline,
                "redecode_answer": redecode,
                "self_critique": self_critique,
                "self_revised_answer": self_revised,
                "external_critique": external_critique,
                "external_revised_answer": external_revised,
                "baseline_correct": int(b_score),
                "redecode_correct": int(r_score),
                "self_revised_correct": int(s_score),
                "external_revised_correct": e_score,
                "drift_norm_self": drift_norm,
                "was_corrected_by_self": int((b_score == 0) and (s_score == 1)),
                "remained_wrong_after_self": int((b_score == 0) and (s_score == 0)),
            }
        )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")

    df = pd.DataFrame(records)
    cos_df = pd.DataFrame(layer_cos_rows)
    l2_df = pd.DataFrame(layer_l2_rows)

    df.to_csv(eval_dir / "example_level_results.csv", index=False)
    cos_df.to_csv(eval_dir / "layer_cosine_distances.csv", index=False)
    l2_df.to_csv(eval_dir / "layer_l2_distances.csv", index=False)

    # Behavioral stats.
    baseline_acc = float(df["baseline_correct"].mean())
    redecode_acc = float(df["redecode_correct"].mean())
    self_acc = float(df["self_revised_correct"].mean())
    ext_df = df.dropna(subset=["external_revised_correct"])
    ext_acc = float(ext_df["external_revised_correct"].mean()) if len(ext_df) else math.nan

    delta_self = df["self_revised_correct"].to_numpy() - df["baseline_correct"].to_numpy()
    delta_red = df["redecode_correct"].to_numpy() - df["baseline_correct"].to_numpy()
    ci_self = bootstrap_ci(delta_self)
    ci_red = bootstrap_ci(delta_red)

    mcnemar_self = mcnemar(
        [
            [
                int(((df["baseline_correct"] == 0) & (df["self_revised_correct"] == 0)).sum()),
                int(((df["baseline_correct"] == 0) & (df["self_revised_correct"] == 1)).sum()),
            ],
            [
                int(((df["baseline_correct"] == 1) & (df["self_revised_correct"] == 0)).sum()),
                int(((df["baseline_correct"] == 1) & (df["self_revised_correct"] == 1)).sum()),
            ],
        ],
        exact=False,
        correction=True,
    )

    # Drift stats by layer.
    layer_stats = cos_df.groupby(["layer", "condition"], as_index=False).agg(
        mean=("cosine_distance", "mean"),
        std=("cosine_distance", "std"),
        count=("cosine_distance", "count"),
    )
    layer_stats.to_csv(eval_dir / "layer_drift_summary.csv", index=False)

    # Per-layer paired tests: self_critique vs redecode.
    pvals = []
    test_rows = []
    for layer in sorted(cos_df["layer"].unique()):
        sub = cos_df[cos_df["layer"] == layer]
        pivot = sub.pivot_table(index="example_id", columns="condition", values="cosine_distance", aggfunc="mean")
        if {"self_critique", "redecode"}.issubset(pivot.columns):
            x = pivot["self_critique"].to_numpy()
            y = pivot["redecode"].to_numpy()
            stat = wilcoxon(x, y, zero_method="wilcox", correction=True, alternative="greater")
            pvals.append(stat.pvalue)
            test_rows.append(
                {
                    "layer": int(layer),
                    "wilcoxon_stat": float(stat.statistic),
                    "p_value": float(stat.pvalue),
                    "mean_self": float(np.mean(x)),
                    "mean_redecode": float(np.mean(y)),
                    "cohens_d_paired": cohens_d_paired(x, y),
                }
            )
    if test_rows:
        reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        for row, pa, rj in zip(test_rows, p_adj, reject, strict=True):
            row["p_adj_fdr_bh"] = float(pa)
            row["reject_fdr_0_05"] = bool(rj)
    layer_test_df = pd.DataFrame(test_rows)
    layer_test_df.to_csv(eval_dir / "layer_wilcoxon_tests.csv", index=False)

    # Probe analysis: corrected vs remained wrong, using final-layer drift vector.
    drift_feats = []
    labels = []
    for _, row in df.iterrows():
        if row["was_corrected_by_self"] == 1 or row["remained_wrong_after_self"] == 1:
            ex_id = row["example_id"]
            layer_sub = cos_df[(cos_df["example_id"] == ex_id) & (cos_df["condition"] == "self_critique")]
            # Feature: per-layer cosine drift profile.
            feat = layer_sub.sort_values("layer")["cosine_distance"].to_numpy()
            drift_feats.append(feat)
            labels.append(int(row["was_corrected_by_self"]))

    probe_metrics = {"n_probe_examples": len(labels), "positive_rate": float(np.mean(labels) if labels else 0.0)}
    if len(labels) >= 20 and len(set(labels)) == 2:
        X = np.vstack(drift_feats)
        y = np.array(labels)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        aucs = []
        f1s = []
        for tr, te in cv.split(X, y):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[tr], y[tr])
            proba = clf.predict_proba(X[te])[:, 1]
            pred = (proba >= 0.5).astype(int)
            aucs.append(roc_auc_score(y[te], proba))
            f1s.append(f1_score(y[te], pred))
        probe_metrics.update(
            {
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs, ddof=1)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s, ddof=1)),
            }
        )
    else:
        probe_metrics["note"] = "Insufficient corrected/uncorrected examples for stable CV probe."

    # Drift stability among corrected examples: pairwise cosine between drift profiles.
    corr_df = df[df["was_corrected_by_self"] == 1]
    corrected_profiles = []
    for ex_id in corr_df["example_id"].tolist():
        vec = cos_df[(cos_df["example_id"] == ex_id) & (cos_df["condition"] == "self_critique")].sort_values("layer")[
            "cosine_distance"
        ].to_numpy()
        if len(vec):
            corrected_profiles.append(vec)
    pairwise = []
    for i in range(len(corrected_profiles)):
        for j in range(i + 1, len(corrected_profiles)):
            a = corrected_profiles[i]
            b = corrected_profiles[j]
            pairwise.append(1.0 - cosine_distance(a, b))
    drift_stability = float(np.mean(pairwise)) if pairwise else math.nan

    summary = {
        "config": {
            "seed": args.seed,
            "local_model": args.local_model,
            "external_model": args.external_model if ext_client else None,
            "n_examples_total": len(df),
            "n_gsm8k": args.n_gsm8k,
            "n_csqa": args.n_csqa,
            "n_external_calls": ext_calls,
            "max_new_tokens": args.max_new_tokens,
            "mixed_precision": bool(torch.cuda.is_available()),
            "device": str(local_llm.device),
        },
        "behavior": {
            "baseline_accuracy": baseline_acc,
            "redecode_accuracy": redecode_acc,
            "self_revised_accuracy": self_acc,
            "external_revised_accuracy": ext_acc,
            "delta_self_mean": float(np.mean(delta_self)),
            "delta_redecode_mean": float(np.mean(delta_red)),
            "delta_self_95ci": ci_self,
            "delta_redecode_95ci": ci_red,
            "mcnemar_self_vs_baseline_stat": float(mcnemar_self.statistic),
            "mcnemar_self_vs_baseline_p": float(mcnemar_self.pvalue),
        },
        "representation": {
            "mean_cosine_drift_self": float(cos_df[cos_df["condition"] == "self_critique"]["cosine_distance"].mean()),
            "mean_cosine_drift_redecode": float(cos_df[cos_df["condition"] == "redecode"]["cosine_distance"].mean()),
            "drift_stability_corrected_pairwise_cosine": drift_stability,
        },
        "probe": probe_metrics,
    }

    with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    # Plots
    sns.set_theme(style="whitegrid")

    acc_df = pd.DataFrame(
        [
            {"condition": "baseline", "accuracy": baseline_acc},
            {"condition": "redecode", "accuracy": redecode_acc},
            {"condition": "self_critique", "accuracy": self_acc},
            {"condition": "external_critique", "accuracy": ext_acc},
        ]
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=acc_df, x="condition", y="accuracy", hue="condition", palette="Set2", legend=False)
    plt.ylim(0, 1.0)
    plt.title("Accuracy by Condition")
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_by_condition.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=layer_stats, x="layer", y="mean", hue="condition", marker="o")
    plt.title("Residual Drift by Layer (Cosine Distance from Baseline)")
    plt.ylabel("Mean cosine distance")
    plt.tight_layout()
    plt.savefig(plots_dir / "drift_by_layer.png", dpi=160)
    plt.close()

    corr_only = df[df["baseline_correct"] == 0].copy()
    if len(corr_only):
        corr_only["self_fixed"] = corr_only["self_revised_correct"].astype(int)
        plt.figure(figsize=(7, 4))
        sns.countplot(data=corr_only, x="self_fixed", hue="self_fixed", palette="Set1", legend=False)
        plt.title("Initially Wrong Examples: Self-Critique Outcome")
        plt.xlabel("Corrected (1) vs Still Wrong (0)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots_dir / "correction_outcomes.png", dpi=160)
        plt.close()

    print(json.dumps(to_jsonable(summary), indent=2))
    print(f"Saved outputs under: {results_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Representation drift under self-critique experiments.")
    parser.add_argument("--workspace-root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--external-model", type=str, default="gpt-4.1")
    parser.add_argument("--n-gsm8k", type=int, default=40)
    parser.add_argument("--n-csqa", type=int, default=40)
    parser.add_argument("--n-external", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--use-external", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
