# Resources Catalog

## Summary
This document catalogs papers, datasets, and code resources gathered for the project **Representation Drift Under Self-Reflection**.

### Counts
- Papers downloaded: 12
- Datasets downloaded: 3
- Repositories cloned: 5

## Papers

| Title | Authors | Year | File | Key Info |
|---|---|---:|---|---|
| Self-Refine: Iterative Refinement with Self-Feedback | Madaan et al. | 2023 | `papers/2303.17651_...pdf` | Iterative self-feedback baseline |
| Reflexion: Language Agents with Verbal Reinforcement Learning | Shinn et al. | 2023 | `papers/2303.11366_...pdf` | Reflection memory for retries |
| CRITIC | Gou et al. | 2023 | `papers/2305.11738_...pdf` | Tool-grounded critique |
| LLMs Cannot Self-Correct Reasoning Yet | Huang et al. | 2023 | `papers/2310.01798_...pdf` | Intrinsic correction limits |
| LLMs cannot find reasoning errors... | Jin et al. | 2024 | `papers/acl2024-findings-826_...pdf` | Error localization bottleneck |
| LLMs have Intrinsic Self-Correction Ability | Liu et al. | 2024 | `papers/2406.15673_...pdf` | Controlled prompt/temperature gains |
| Analysing the Residual Stream Under Knowledge Conflicts | Zhao et al. | 2024 | `papers/2410.16090_...pdf` | Residual-stream conflict probing |
| Inspecting and Editing Knowledge Representations | Hernandez et al. | 2023 | `papers/2304.00740_...pdf` | Probe/intervention methods |
| Representation Engineering | Zou et al. | 2023 | `papers/2310.01405_...pdf` | Representation-level pipelines |
| Activation Engineering | Turner et al. | 2023 | `papers/2308.10248_...pdf` | Activation-addition steering |
| Refusal Is Mediated by a Single Direction | Arditi et al. | 2024 | `papers/2406.11717_...pdf` | Low-dimensional behavior direction |
| Reasoning via Steering Vectors | Venhoff et al. | 2025 | `papers/2506.18167_...pdf` | Reasoning control with vectors |

See `papers/README.md` and `papers/paper_metadata.json` for details.

## Datasets

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| gsm8k (main) | HuggingFace | train 7,473 / test 1,319 | math reasoning | `datasets/gsm8k/` | verifiable answers |
| commonsense_qa | HuggingFace | train 9,741 / val 1,221 / test 1,140 | commonsense MCQ | `datasets/commonsense_qa/` | non-math reasoning |
| truthful_qa (multiple_choice) | HuggingFace | val 817 | truthfulness | `datasets/truthful_qa_multiple_choice/` | hallucination-sensitive |

See `datasets/README.md` and `datasets/dataset_summary.json`.

## Code Repositories

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| self-refine | https://github.com/madaan/self-refine | iterative self-feedback | `code/self-refine/` | includes GSM workflow |
| reflexion | https://github.com/noahshinn/reflexion | verbal RL reflection agents | `code/reflexion/` | multi-attempt memory loops |
| representation-engineering | https://github.com/andyzoujm/representation-engineering | representation probing/control | `code/representation-engineering/` | direct fit to drift analysis |
| activation_additions | https://github.com/montemac/activation_additions | activation vector injection | `code/activation_additions/` | residual-stream interventions |
| llm-attacks | https://github.com/llm-attacks/llm-attacks | robustness/adversarial perturbation infra | `code/llm-attacks/` | optional stress testing |

See `code/README.md`.

## Resource Gathering Notes

### Search Strategy
1. Ran paper-finder in diligent mode on the exact hypothesis topic.
2. Filtered by direct topical fit (self-critique + internal representations).
3. Added targeted arXiv/ACL retrieval for missing but central works.
4. Prioritized papers with methods/code supporting activation-level experiments.

### Selection Criteria
- Direct relevance to self-correction or internal representation analysis.
- Availability of downloadable PDF and implementation support.
- Benchmark compatibility with verifiable reasoning tasks.

### Challenges Encountered
- Paper-finder output contained many low-fit/noisy entries.
- Top paper-finder item had no OA PDF; another was API-rate-limited.
- Some repositories require paid APIs or large GPUs for full reruns.

### Gaps and Workarounds
- Limited direct prior work combining reflection loops + residual drift metrics.
- Workaround: combine self-correction frameworks (Self-Refine/Reflexion) with RepE/activation-steering toolchains.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: `gsm8k` + `commonsense_qa`; add `truthful_qa` for robustness checks.
2. **Baseline methods**: one-shot CoT, Self-Refine, Reflexion, oracle-feedback upper bound.
3. **Evaluation metrics**: accuracy delta, probe AUROC/AUPRC, drift geometry (cosine/CKA/PCA).
4. **Code to adapt/reuse**: start from `representation-engineering`, `activation_additions`, and `self-refine`.

## Execution Update (2026-03-09)

Automated experiment execution completed in this workspace with the following new artifacts:
- Script: `src/run_representation_drift_experiment.py`
- Plan: `planning.md`
- Report: `REPORT.md`
- Results: `results/evaluations/` and `results/plots/`

Main run configuration:
- Local model: `Qwen/Qwen2.5-1.5B-Instruct`
- External critique model: `gpt-4.1` (subset)
- Data subset: 24 GSM8K + 24 CommonsenseQA
- Seed: 42

This extends the pre-gathered resources with actual empirical outputs for the representation-drift hypothesis.
