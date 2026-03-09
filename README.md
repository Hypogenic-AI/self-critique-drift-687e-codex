# Representation Drift Under Self-Reflection

This project tests whether self-critique in LLMs creates meaningful internal representation change (residual-stream drift) versus shallow re-sampling. We run controlled prompting experiments with activation capture and statistical analysis on GSM8K and CommonsenseQA.

## Key Findings
- On `Qwen/Qwen2.5-1.5B-Instruct` (48 examples), self-critique reduced accuracy from **39.6% to 16.7%**.
- Re-decode control also reduced accuracy (**39.6% -> 25.0%**), indicating second-pass instability.
- Residual drift under self-critique was very similar to re-decode (mean cosine drift **0.5700 vs 0.5719**).
- Layer-wise self-vs-redecode drift differences were not significant after FDR correction.
- External critique via GPT-4.1 (subset n=16) matched intrinsic self-critique (both 25.0%) and remained below baseline.

See [REPORT.md](./REPORT.md) for full methodology, statistics, figures, and limitations.

## Reproduce
```bash
# 1) Activate environment
source .venv/bin/activate

# 2) Run experiment (main configuration)
python src/run_representation_drift_experiment.py \
  --workspace-root . \
  --n-gsm8k 24 \
  --n-csqa 24 \
  --n-external 16 \
  --use-external \
  --max-new-tokens 96
```

Notes:
- Requires `OPENAI_API_KEY` for external critique condition.
- Without API key, omit `--use-external`.

## File Structure
- `planning.md`: planning, motivation, novelty, and experiment design.
- `src/run_representation_drift_experiment.py`: end-to-end experiment pipeline.
- `results/evaluations/metrics.json`: primary metrics summary.
- `results/evaluations/*.csv`: per-example and per-layer analysis tables.
- `results/plots/*.png`: generated visualizations.
- `REPORT.md`: comprehensive research report.
