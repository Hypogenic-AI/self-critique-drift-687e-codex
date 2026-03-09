# Outline: Representation Drift Under Self-Reflection

## Abstract
- Problem: whether self-critique reshapes internal states or only resamples outputs.
- Protocol: baseline vs re-decode control vs self-critique; residual drift analysis.
- Main results: accuracy drops and near-identical drift magnitudes.
- Significance: caution for reflection prompting in weak-model regimes.

## 1. Introduction
- Hook: reflection is widely deployed but mechanistic evidence is thin.
- Gap: prior work separates output metrics from internal-state geometry.
- Approach: unified behavior + residual-stream drift protocol.
- Preview: large accuracy drop and no unique self-critique drift signature.
- Contributions: protocol, controlled comparison, statistical analysis, failure characterization.

## 2. Related Work
- Reflection and self-correction methods.
- Intrinsic vs oracle/tool-guided correction.
- Representation analysis and activation engineering.
- Positioning: bridge behavior and internal geometry under one protocol.

## 3. Methodology
- Problem setup and conditions.
- Datasets, prompts, scoring rules.
- Activation extraction and drift metrics.
- Statistical tests and probing setup.
- Implementation and reproducibility details.

## 4. Results
- Main condition-level accuracies and dataset-wise breakdown.
- Transition analysis and significance tests.
- Representation drift comparison and layer-wise tests.
- External critique subset outcome.
- Probe instability due severe class imbalance.

## 5. Discussion
- Interpretation: destabilizing second-pass generation.
- Why self-critique fails in this regime.
- Limitations and implications.

## 6. Conclusion
- Summary and key takeaway.
- Future work directions.
