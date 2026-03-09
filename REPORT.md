# Representation Drift Under Self-Reflection: Does Self-Critique Reshape Internal States?

## 1. Executive Summary
We tested whether self-critique causes structured internal representation change (residual-stream drift) versus shallow re-sampling.

Across 48 examples (24 GSM8K + 24 CommonsenseQA) on `Qwen/Qwen2.5-1.5B-Instruct`, self-critique **reduced** accuracy relative to baseline (39.6% -> 16.7%), and re-decode control also reduced accuracy (39.6% -> 25.0%). Residual drift magnitude under self-critique was almost identical to re-decode control (mean cosine drift 0.5700 vs 0.5719; paired Wilcoxon p=0.764 on per-example mean drift), indicating no clear evidence of uniquely structured reflection-induced drift in this setup.

Practical implication: for this model/task/prompt regime, self-reflection mostly behaved like destabilizing re-generation rather than meaningful internal correction.

## 2. Goal
### Hypothesis
Self-critique induces structured residual-stream shifts associated with improved reasoning and better separability of corrected trajectories.

### Why Important
If true, self-reflection prompts can be interpreted as internal state shaping rather than output-level prompt tricks. This matters for mechanistic interpretability, recursive AI agents, and reliability-oriented prompting.

### Problem Addressed
The field has output-level evidence for/against self-correction, but weaker causal/representational evidence linking reflection to internal geometry changes.

## 3. Data Construction

### Dataset Description
- `gsm8k` test split (sampled 24 examples): arithmetic word-problem reasoning with numeric final answers.
- `commonsense_qa` validation split (sampled 24 examples): multiple-choice commonsense reasoning.
- Local sources: `datasets/gsm8k/`, `datasets/commonsense_qa/`.

### Example Samples
| Dataset | Example (abridged) | Label |
|---|---|---|
| GSM8K | “A toy costs $18 and is discounted 25%... what is the final price?” | Numeric answer text |
| GSM8K | “A class has 28 students, 3/7 are absent... how many present?” | Numeric answer text |
| CommonsenseQA | “What would you do before mailing a letter?” with options A-E | `answerKey` (A-E) |

### Data Quality
- Missing values: 0% observed in sampled fields (`question`, labels, options)
- Duplicates: none observed in sampled IDs
- Class/task distribution in experiment: 50% GSM8K, 50% CommonsenseQA
- Validation checks: schema checks, sampled prompt formatting checks, score parsing checks

### Preprocessing Steps
1. Sampled examples with fixed seed (`42`) from official local HF disk datasets.
2. Built task-specific prompts with strict final-answer formats.
3. For scoring:
   - GSM8K: parsed last numeric token from model output and gold rationale text.
   - CSQA: parsed A-E letter from final output line.
4. Generated condition outputs: baseline, re-decode control, self-critique revise, and (subset) external critique revise.
5. Extracted per-layer residual representations using final-token hidden states of prompt+answer text.

### Train/Val/Test Splits
- No model training/fine-tuning.
- Evaluation-only protocol on sampled benchmark subsets.
- Probe model used cross-validation on trajectory-level features.

## 4. Experiment Description

### Methodology
#### High-Level Approach
Compare behavioral changes and internal drift jointly across conditions:
- Baseline answer
- Re-decode without critique (control)
- Intrinsic self-critique + revision
- External critique (GPT-4.1) + local revision (subset)

#### Why This Method
This directly separates “any second attempt” effects from critique-conditioned effects while preserving activation access in a local model.

### Implementation Details
#### Tools and Libraries
- Python 3.12.8
- torch 2.10.0+cu128
- transformers 5.3.0
- datasets 4.6.1
- scikit-learn 1.8.0
- scipy 1.17.1
- statsmodels 0.14.6
- openai 2.26.0

#### Algorithms/Models
- Local model: `Qwen/Qwen2.5-1.5B-Instruct`
- External critique model: `gpt-4.1` API (16 critique calls)
- Probe classifier: logistic regression on per-layer cosine-drift profiles

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---:|---|
| seed | 42 | fixed reproducibility |
| n_gsm8k | 24 | runtime-feasible sample |
| n_csqa | 24 | runtime-feasible sample |
| n_external | 16 | cost/runtime constrained |
| max_new_tokens | 96 | prompt-format sufficiency |
| decoding temperature | 0.0 | deterministic control |
| probe CV folds | 5 requested | default (later limited by class imbalance) |

#### Pipeline
1. Load datasets and sample examples.
2. Generate baseline answer.
3. Generate re-decode control answer.
4. Generate intrinsic critique and revised answer.
5. For subset: request GPT-4.1 critique and regenerate revised answer.
6. Extract layer-wise residual vectors (last token per layer).
7. Compute drift metrics and behavior deltas.
8. Run statistical tests and generate plots.

### Experimental Protocol
#### Reproducibility Information
- Runs averaged: 1 full run + 1 smoke run
- Seed: 42
- Hardware: 2x NVIDIA RTX 3090 (24GB each), CUDA enabled
- Mixed precision: enabled (`torch.autocast` bfloat16)
- Effective generation batch style: sequential (activation capture dominated); for 24GB VRAM this is safer than large-batch generation for long outputs
- Execution time: ~13 minutes for the 48-example main run

#### Evaluation Metrics
- Accuracy by condition
- Accuracy delta vs baseline with bootstrap 95% CI
- McNemar paired significance test (baseline vs self-revised)
- Layer-wise cosine drift and L2 drift from baseline state
- Wilcoxon signed-rank tests per layer (self-critique vs re-decode), BH-FDR correction
- Probe separability metrics (AUC/F1) for corrected vs uncorrected trajectories

### Raw Results
#### Main Performance Table
| Condition | Accuracy |
|---|---:|
| Baseline | 0.3958 |
| Re-decode control | 0.2500 |
| Self-critique revised | 0.1667 |
| External-critique revised (n=16 subset) | 0.2500 |

#### Statistical Summary
| Metric | Value |
|---|---:|
| Self - Baseline delta | -0.2292 |
| 95% CI (bootstrap) | [-0.3750, -0.0625] |
| Re-decode - Baseline delta | -0.1458 |
| 95% CI (bootstrap) | [-0.2708, -0.0208] |
| McNemar p (Self vs Baseline) | 0.0098 |
| Mean cosine drift (Self) | 0.5700 |
| Mean cosine drift (Re-decode) | 0.5719 |
| Paired Wilcoxon p on per-example mean drift | 0.7643 |

#### Dataset-Level Accuracy
| Dataset | Baseline | Re-decode | Self-critique |
|---|---:|---:|---:|
| CommonsenseQA (n=24) | 0.7917 | 0.5000 | 0.2500 |
| GSM8K (n=24) | 0.0000 | 0.0000 | 0.0833 |

#### Transition Counts (Baseline -> Self)
- Improved: 2
- Degraded: 13
- Unchanged: 33

#### Probe/Separability
- Probe set size: 29 (initially wrong examples)
- Positive class (corrected): 2 (6.9%)
- Result: severe imbalance caused unstable AUC; CV AUC undefined in several folds, F1 mean 0.0.

#### Output Locations
- Results JSON: `results/evaluations/metrics.json`
- Tables/CSVs: `results/evaluations/`
- Plots: `results/plots/`

## 5. Result Analysis

### Key Findings
1. **Self-critique harmed accuracy** in this run (39.6% -> 16.7%, significant by McNemar p=0.0098).
2. **Re-decode control also harmed accuracy** (39.6% -> 25.0%), suggesting second-pass generation instability.
3. **Representation drift magnitude was not larger for self-critique** than re-decode (0.5700 vs 0.5719; negligible effect size).
4. **Layer-wise drift differences were not significant after FDR correction** across all 28 layers.
5. **External critique did not outperform intrinsic self-critique on subset** (25.0% vs 25.0%) and remained below baseline subset accuracy (37.5%).

### Hypothesis Testing Results
- H0 (behavior): self-critique does not improve accuracy over baseline.
- H1 (behavior): self-critique improves accuracy.
- Outcome: reject H1 for this setup; evidence supports degradation, not improvement.

- H0 (representation): self-critique drift equals re-decode drift.
- H1 (representation): self-critique drift is larger/more structured.
- Outcome: fail to reject H0 (Wilcoxon p=0.764 on per-example layer-mean drift; per-layer tests also non-significant after FDR).

### Comparison to Baselines
- Self-critique underperformed baseline by 22.9 absolute points.
- Re-decode underperformed baseline by 14.6 points.
- External critique matched self-critique on the evaluated subset and did not recover baseline performance.

### Visualizations
- `results/plots/accuracy_by_condition.png`: condition-level performance drop under reflection.
- `results/plots/drift_by_layer.png`: similar drift profiles for self-critique and re-decode across layers.
- `results/plots/correction_outcomes.png`: very low correction count among initially wrong items.

### Surprises and Insights
- Contrary to expected “meaningful reflection,” intrinsic critique frequently degraded already-correct CSQA answers.
- GSM8K baseline near-zero suggests local model capacity/format mismatch dominates this task slice.
- Drift profile similarity implies critique text alone did not induce qualitatively distinct internal movement versus a simple second try.

### Error Analysis
- Common failure mode: revision introduces new mistakes after valid initial answer.
- Weak error localization: critique text often generic (“double-check”) rather than specific computational fault localization.
- Severe class imbalance (2 corrected vs 27 still wrong among initially wrong) prevented stable separability conclusions.

### Limitations
- Sample size is moderate (48), not full-benchmark scale.
- Single local model and single prompt template family.
- Probe analysis underpowered due low correction prevalence.
- Internal representation approximation used final-token layer states, not full trajectory token-level dynamics.
- External critique condition used subset (n=16) for budget/runtime reasons.

## 6. Conclusions
In this experiment, self-critique did not show evidence of beneficial or uniquely structured residual-stream change. Behavioral outcomes degraded significantly, and internal drift closely matched a non-critique re-decode control. This supports the “shallow re-sampling / destabilization” interpretation for this particular model-task-prompt regime.

Confidence is moderate for the negative behavioral result and moderate-low for separability conclusions (class imbalance). Stronger claims require larger samples, stronger base models, and richer activation trajectory features.

## 7. Next Steps
### Immediate Follow-ups
1. Repeat with a stronger local model (e.g., 7B+) while keeping identical protocol.
2. Add explicit error-location scaffolds before revision to test the localization bottleneck hypothesis.
3. Use multi-round reflection (2-3 rounds) with stop criteria and compare to single-round.

### Alternative Approaches
- Token-level trajectory drift (not only final-token state).
- CKA/CCA subspace metrics across full hidden-state sequences.
- Contrastive probing on critique text quality (specific vs generic critiques).

### Broader Extensions
- Human critique vs self critique vs model-API critique with matched style and length.
- Cross-model scaling study (1.5B, 7B, API frontier models).
- Agent-loop settings where reflection affects tool calls and planning states.

### Open Questions
- Are structured reflection subspaces only visible when the model is already competent on the base task?
- Does critique usefulness require external grounding signals (tools, verifiers) to produce distinct internal geometry?
- Which layers/tokens are most sensitive to useful vs harmful reflection prompts?

## References
Primary references are listed in `literature_review.md` and `resources.md`, including:
- Self-Refine (Madaan et al., 2023)
- Reflexion (Shinn et al., 2023)
- LLMs Cannot Self-Correct Reasoning Yet (Huang et al., 2023)
- LLMs have Intrinsic Self-Correction Ability (Liu et al., 2024)
- Analysing the Residual Stream Under Knowledge Conflicts (Zhao et al., 2024)
- Representation Engineering (Zou et al., 2023)
