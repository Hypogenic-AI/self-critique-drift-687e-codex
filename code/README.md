# Cloned Repositories

## 1) self-refine
- URL: https://github.com/madaan/self-refine
- Purpose: Iterative self-feedback and refinement framework.
- Location: `code/self-refine/`
- Key files:
  - `src/gsm/run.py`
  - `src/*/run.py` task pipelines
- Notes:
  - Requires LLM API setup.
  - Includes GSM8K and other task examples for iterative refinement.

## 2) reflexion
- URL: https://github.com/noahshinn/reflexion
- Purpose: Reflexion agents with verbal RL memory.
- Location: `code/reflexion/`
- Key files:
  - `hotpotqa_runs/`
  - `alfworld_runs/`
  - `programming_runs/`
- Notes:
  - Uses OpenAI API key and environment-specific dependencies.
  - Good baseline for multi-round reflection trajectories.

## 3) representation-engineering
- URL: https://github.com/andyzoujm/representation-engineering
- Purpose: RepE/RepReading/RepControl for representation-level analysis and steering.
- Location: `code/representation-engineering/`
- Key files:
  - `repe/`
  - `examples/`
  - `repe_eval/`
- Notes:
  - Strong fit for probing/steering residual-space directions.
  - Install via `pip install -e .` in repo.

## 4) activation_additions
- URL: https://github.com/montemac/activation_additions
- Purpose: Activation-vector injection and algebraic value editing.
- Location: `code/activation_additions/`
- Key files:
  - `scripts/basic_functionality.py`
  - `activation_additions/`
- Notes:
  - Uses hooks into residual stream (TransformerLens style).
  - Useful for direction-based intervention baselines.

## 5) llm-attacks
- URL: https://github.com/llm-attacks/llm-attacks
- Purpose: Optimization-based activation/prompt attack infrastructure.
- Location: `code/llm-attacks/`
- Key files:
  - `experiments/`
  - `llm_attacks/`
- Notes:
  - Not a direct reflection baseline, but useful for stress-testing robustness under perturbation.
  - Typically expects high-memory GPUs and local model weights.

## Validation status
- Repositories cloned successfully.
- Deep run tests were not executed due API/GPU/runtime requirements.
- For experiment runner: start with `representation-engineering`, `activation_additions`, and `self-refine`.
