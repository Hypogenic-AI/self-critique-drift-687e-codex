# Planning: Representation Drift Under Self-Reflection

## Motivation & Novelty Assessment

### Why This Research Matters
Self-critique prompting is increasingly used in production and research systems to improve reasoning quality, but output gains alone do not establish that models are performing deeper internal correction. Measuring whether reflection changes internal residual-stream geometry matters for mechanistic interpretability, safer recursive self-improvement loops, and principled agent design. A reliable metric of internal “cognitive change” can distinguish true revision from shallow re-sampling.

### Gap in Existing Work
From `literature_review.md`, prior work either (a) studies self-correction mostly at output level (e.g., Self-Refine/Reflexion; intrinsic-vs-oracle correction papers) or (b) studies residual-stream structure without reflection loops (representation engineering/conflict probing). There is limited work jointly quantifying reflection-induced representation drift, geometric structure, and accuracy deltas in one controlled protocol.

### Our Novel Contribution
We run a unified protocol combining behavioral self-critique with residual-stream analysis on the same examples: (1) pre/post reflection accuracy, (2) residual drift magnitude and direction stability, and (3) probe separability of corrected vs uncorrected trajectories. We additionally contrast intrinsic self-critique with external critique to test whether stronger supervision creates more structured drift.

### Experiment Justification
- Experiment 1: Baseline vs self-critique behavior on GSM8K/CommonsenseQA. Why needed: establishes whether reflection improves outputs in our setup and identifies corrected/failure subsets for internal analysis.
- Experiment 2: Residual-stream drift quantification across rounds/layers. Why needed: directly tests structured internal change vs shallow local perturbation.
- Experiment 3: Linear separability/probe study (corrected vs uncorrected trajectories). Why needed: evaluates whether reflection creates more distinct reasoning subspaces.
- Experiment 4: Intrinsic self-critique vs external critique comparison. Why needed: tests whether critique source changes drift structure and stability.
- Experiment 5: Robustness checks (multiple seeds, single vs multi-round). Why needed: ensures effects are stable rather than decoding noise.

## Research Question
Does self-critique induce structured residual-stream representation shifts associated with improved reasoning, or are improvements primarily shallow re-sampling with minimal internal geometric change?

## Background and Motivation
Recent reflection methods show mixed behavioral results: some report gains under controlled prompting, while others show intrinsic self-correction often fails without external guidance. Mechanistic work demonstrates that behavior and knowledge conflict signals are encoded in low-dimensional residual directions, but has not tightly linked these structures to reflection loops. This project bridges those lines by jointly measuring behavior and internal state dynamics.

## Hypothesis Decomposition
- H1 (behavior): Self-critique increases final-answer accuracy over one-shot baseline.
- H2 (drift): Reflection rounds induce larger and more layer-structured residual drift than no-reflection re-decoding.
- H3 (geometry): Corrected trajectories are more linearly separable from uncorrected trajectories than baseline trajectories.
- H4 (stability): Drift vectors for successful correction show higher cross-example alignment (subspace stability).
- Alternative explanation A: Improvements are sampling artifacts; drift remains near baseline neighborhood.
- Alternative explanation B: Drift exists but is unstructured noise (low separability, low alignment).

## Proposed Methodology

### Approach
Use a local open model for internal activation access (`Qwen/Qwen2.5-1.5B-Instruct`) with deterministic decoding and standardized prompts. For each example, run: (a) baseline answer, (b) self-critique+revision, (c) optional external critique+revision. Capture residual hidden states at each layer for final-token representation of each stage. Compute drift and probe metrics; test significance with paired tests and bootstrap CIs.

### Experimental Steps
1. Load datasets (GSM8K, CommonsenseQA) and sample balanced subsets for feasibility.
2. Run baseline prompting and score correctness.
3. Run intrinsic self-critique and revision, then rescore.
4. Run external critique condition (structured critique template) and revision.
5. Extract per-layer residual representations for baseline/critique/revision outputs.
6. Compute drift metrics: cosine distance, L2 norm, and inter-round angular consistency.
7. Train linear probes (logistic regression) to classify corrected vs uncorrected trajectories from residual features.
8. Perform statistical tests: paired t-test/Wilcoxon (behavior), permutation test for probe AUC differences, bootstrap CIs for drift differences.
9. Run robustness checks across random seeds and number of critique rounds.

### Baselines
- One-shot direct reasoning answer (no reflection).
- Re-decode control (second answer request without critique instruction).
- Intrinsic self-critique + revise.
- External critique + revise (pseudo-oracle structured guidance).

### Evaluation Metrics
- Accuracy and accuracy delta (post - pre).
- Correction rate among initially wrong examples.
- Residual drift magnitude (cosine distance, L2) by layer.
- Drift stability (mean pairwise cosine among successful-correction drift vectors).
- Probe separability: AUROC, accuracy, F1 for corrected vs uncorrected labels.

### Statistical Analysis Plan
- Significance level: alpha = 0.05.
- Behavior comparisons: paired bootstrap CIs + McNemar test where applicable.
- Drift comparisons across conditions: paired Wilcoxon signed-rank tests (non-normal robust).
- Probe metric uncertainty: bootstrap over examples (1,000 resamples).
- Multiple comparisons across layers: Benjamini-Hochberg FDR correction.
- Effect sizes: Cohen’s d (paired) and Cliff’s delta for non-parametric contrasts.

## Expected Outcomes
Support hypothesis if we observe: (1) significant accuracy gain after critique, (2) larger structured drift under critique than re-decode control, (3) higher probe separability for corrected trajectories, and (4) consistent drift directions among successful corrections. Refute if gains are absent or drift is minimal/unstructured.

## Timeline and Milestones
- Milestone 1 (setup + EDA): 20 min.
- Milestone 2 (implementation): 60 min.
- Milestone 3 (experiments): 60 min.
- Milestone 4 (analysis + report): 45 min.
- Buffer/debugging: 30 min.

## Potential Challenges
- API key unavailable for external closed-model validation. Mitigation: run primary internal analysis with open model and document API check outcome.
- Runtime constraints for large models. Mitigation: use 1.5B model, capped sample size, and GPU mixed precision.
- Prompt sensitivity. Mitigation: deterministic decoding + seed sweeps + control prompts.

## Success Criteria
- Complete end-to-end reproducible pipeline producing saved metrics and plots.
- At least 100 evaluated examples across tasks with activation captures.
- Statistical tests and effect sizes reported for key hypotheses.
- REPORT.md includes objective findings, limitations, and reproducibility details.
