# Literature Review: Representation Drift Under Self-Reflection

## Review Scope

### Research Question
Does self-critique in LLMs induce structured internal representation drift (especially in the residual stream) that corresponds to improved reasoning quality, rather than shallow output-level rephrasing?

### Inclusion Criteria
- Studies on self-correction/reflection in LLM reasoning.
- Studies with internal-state/representation analysis (residual stream, activation directions, probing).
- Benchmarks with verifiable correctness (math, QA, commonsense).
- Methods with code or reproducible experimental setup.

### Exclusion Criteria
- Purely application/domain papers with no reasoning/internal-state analysis.
- Papers without empirical evidence on correction/reflection effects.
- Non-LLM or unrelated modality work.

### Time Frame
- Primary focus: 2023-2025.
- Foundational context when needed.

### Sources
- Paper-finder service output (diligent mode) and Semantic Scholar follow-up.
- arXiv/API manual retrieval.
- ACL Anthology for conference version where needed.
- GitHub repos linked to key methods.

## Search Log

| Date (UTC) | Query | Source | Result Count | Notes |
|---|---|---|---:|---|
| 2026-03-09 | representation drift self-critique large language models | paper-finder | 64 | Top 2 high relevance; many noisy/low-fit hits.
| 2026-03-09 | self-correction + residual stream + activation steering queries | arXiv API | 90+ candidates | Filtered to 12 directly relevant papers.
| 2026-03-09 | targeted title lookup for reasoning error localization | ACL Anthology | 1 | Added Findings ACL 2024 paper.

## Screening Results

| Stage | Count |
|---|---:|
| Initial candidates | 64 (paper-finder) + arXiv candidates |
| Abstract/title screened | 93 arXiv API hits |
| Included full-text PDFs | 12 |
| Deep-read with full chunk pass | 3 |

Deep-read papers (all chunks read):
- `2310.01798` (LLMs Cannot Self-Correct Reasoning Yet)
- `2406.15673` (Intrinsic Self-Correction Ability)
- `2410.16090` (Residual Stream Under Knowledge Conflicts)

## Key Papers

### 1) Self-Refine: Iterative Refinement with Self-Feedback (2023)
- **Contribution**: Generic FEEDBACK->REFINE loop improves outputs across tasks.
- **Methodology**: Single-model iterative critique and revision.
- **Datasets/Tasks**: Includes GSM8K and generation tasks.
- **Results**: Consistent gains over one-shot prompting.
- **Code**: https://github.com/madaan/self-refine
- **Relevance**: Core baseline for reflection behavior and trajectory logging.

### 2) Reflexion: Language Agents with Verbal Reinforcement Learning (2023)
- **Contribution**: Uses linguistic memory/reflections across attempts.
- **Methodology**: Agent retries with explicit reflective memory.
- **Datasets/Tasks**: HotPotQA, AlfWorld, programming.
- **Results**: Better success rates vs non-reflective variants in multi-step settings.
- **Code**: https://github.com/noahshinn/reflexion
- **Relevance**: Strong iterative reflection baseline with trajectory structure.

### 3) CRITIC: Tool-Interactive Critiquing (2023/ICLR 2024)
- **Contribution**: Self-correction improves substantially when external tools provide grounding.
- **Methodology**: Model critiques outputs using calculators/search/external tools.
- **Datasets**: QA, reasoning, and factual tasks.
- **Results**: Gains over intrinsic-only correction in many settings.
- **Code**: included in paper resources.
- **Relevance**: Critical contrast condition: grounded vs intrinsic reflection.

### 4) Large Language Models Cannot Self-Correct Reasoning Yet (2023/ICLR 2024)
- **Contribution**: Intrinsic self-correction frequently fails on reasoning; can degrade accuracy.
- **Methodology**: Oracle vs intrinsic correction comparisons.
- **Datasets**: GSM8K, CommonSenseQA, HotpotQA.
- **Key Result**: Oracle feedback helps; intrinsic multi-round often hurts.
- **Relevance**: Direct test of whether reflection is meaningful vs shallow.

### 5) LLMs cannot find reasoning errors, but can correct them given error location (ACL Findings 2024)
- **Contribution**: Decomposes correction into error localization vs revision.
- **Methodology**: Provide explicit error locations to model.
- **Key Result**: Correction ability exists when localization bottleneck is removed.
- **Relevance**: Suggests where internal drift may fail (detection stage) vs succeed (repair stage).

### 6) Large Language Models have Intrinsic Self-Correction Ability (2024)
- **Contribution**: Reports positive intrinsic correction under controlled settings.
- **Methodology**: Emphasizes prompt fairness and low temperature.
- **Datasets**: CommonSenseQA, GSM8K, MMLU subsets, SVAMP.
- **Key Result**: Intrinsic correction can improve under specific prompting/temperature conditions.
- **Relevance**: Counterpoint to #4; motivates controlled ablations.

### 7) Analysing the Residual Stream of Language Models Under Knowledge Conflicts (2024)
- **Contribution**: Probing reveals conflict signal and source selection in residual stream.
- **Methodology**: Probe hidden/residual/MLP/attention activations across layers.
- **Datasets**: ConflictQA, NQSwap.
- **Key Result**: Intermediate activations predict whether model uses context vs parametric knowledge.
- **Relevance**: Closest direct precedent for representation-drift hypothesis in residual stream.

### 8) Inspecting and Editing Knowledge Representations in Language Models (2023/COLM 2024)
- **Contribution**: Locates/edits knowledge-related representations in activations.
- **Methodology**: Probing and intervention in hidden states.
- **Relevance**: Supports linear separability + causal intervention framing.

### 9) Representation Engineering: A Top-Down Approach to AI Transparency (2023)
- **Contribution**: Representation-level reading/control pipelines (RepReading/RepControl).
- **Methodology**: Direction finding and representation manipulation.
- **Relevance**: Ready-to-use methodology for measuring drift and controlling subspaces.

### 10) Steering Language Models With Activation Engineering (2023)
- **Contribution**: Activation addition steers model behavior without retraining.
- **Methodology**: Add/subtract activation vectors at residual hooks.
- **Relevance**: Experimental tool for testing whether reflection subspaces are causally meaningful.

### 11) Refusal in Language Models Is Mediated by a Single Direction (2024)
- **Contribution**: Demonstrates low-dimensional behavior direction in activation space.
- **Relevance**: Strong evidence that linear subspaces can encode high-level behavior.

### 12) Understanding Reasoning in Thinking Language Models via Steering Vectors (2025)
- **Contribution**: Uses steering vectors to analyze/control reasoning style.
- **Relevance**: Directly aligns with representation drift + reasoning separability analysis.

## Common Methodologies
- **Iterative self-reflection loops**: Self-Refine, Reflexion, intrinsic/extrinsic correction papers.
- **Oracle-vs-intrinsic decomposition**: separates error detection from correction capability.
- **Activation probing**: linear probes across layers/residual stream to predict behavior.
- **Activation intervention/steering**: representation engineering and activation addition.
- **Controlled decoding ablations**: temperature and prompt-template effects.

## Standard Baselines
- One-shot CoT prompting.
- Self-consistency voting.
- Multi-agent debate.
- Oracle feedback/self-correction (upper bound).
- Tool-augmented correction (extrinsic grounding).

## Evaluation Metrics
- **Task accuracy**: GSM8K, CommonSenseQA, TruthfulQA-style metrics.
- **Delta accuracy after reflection**: before vs after critique.
- **Probe metrics**: linear separability (accuracy/AUROC/AUPRC) for state classification.
- **Representation drift**: cosine distance/CKA/PCA trajectory displacement across iterations.
- **Stability metrics**: variance of drift vectors across seeds/prompts.

## Datasets in the Literature
- GSM8K, SVAMP (math reasoning).
- CommonSenseQA (commonsense multiple choice).
- HotpotQA (multi-hop QA).
- ConflictQA, NQSwap (knowledge conflict probes).
- Truthfulness-style QA for hallucination robustness.

## Gaps and Opportunities
1. Few papers jointly measure **performance changes + internal representation drift** in one protocol.
2. Reflection studies often stop at output metrics and do not quantify residual-stream geometry.
3. Limited work on **stability of drift subspaces** across tasks/models.
4. Error localization vs correction is under-linked to internal activations.

## Recommendations for Our Experiment
- **Recommended datasets**:
  - `gsm8k` for verifiable reasoning.
  - `commonsense_qa` for non-math reasoning.
  - `truthful_qa` (multiple_choice) for factual reliability stress tests.
- **Recommended baselines**:
  - One-shot CoT.
  - Self-Refine loop.
  - Reflexion-style memory loop.
  - Oracle feedback upper bound.
  - Tool-grounded critique (CRITIC-style).
- **Recommended metrics**:
  - Accuracy delta (pre/post reflection).
  - Linear probe separability of correct vs incorrect traces by layer.
  - Drift magnitude/direction consistency in residual stream.
- **Methodological considerations**:
  - Fix temperature (include `T=0` ablation).
  - Use matched prompts for fair intrinsic vs extrinsic comparison.
  - Store per-step activations for each reflection round.
  - Separate "error found" from "error fixed" in analysis labels.
