# Downloaded Datasets

This directory contains local benchmark datasets for self-reflection and representation-drift experiments. Data files are excluded from git via `.gitignore`.

## Dataset 1: gsm8k (main)

### Overview
- Source: HuggingFace `gsm8k` (`main` config)
- Size: train 7,473; test 1,319
- Format: HuggingFace DatasetDict saved to disk
- Task: grade-school math reasoning (final answer correctness)
- Splits: train, test
- License: see dataset card on HuggingFace

### Download Instructions
Using HuggingFace (recommended):
```python
from datasets import load_dataset
dataset = load_dataset("gsm8k", "main")
dataset.save_to_disk("datasets/gsm8k")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k")
```

### Sample Data
- `datasets/gsm8k/samples/samples.json`

## Dataset 2: commonsense_qa

### Overview
- Source: HuggingFace `commonsense_qa`
- Size: train 9,741; validation 1,221; test 1,140
- Format: HuggingFace DatasetDict saved to disk
- Task: multiple-choice commonsense reasoning
- Splits: train, validation, test
- License: see dataset card on HuggingFace

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("commonsense_qa")
dataset.save_to_disk("datasets/commonsense_qa")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/commonsense_qa")
```

### Sample Data
- `datasets/commonsense_qa/samples/samples.json`

## Dataset 3: truthful_qa (multiple_choice)

### Overview
- Source: HuggingFace `truthful_qa` (`multiple_choice` config)
- Size: validation 817
- Format: HuggingFace DatasetDict saved to disk
- Task: truthfulness and misconception robustness
- Splits: validation
- License: see dataset card on HuggingFace

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("truthful_qa", "multiple_choice")
dataset.save_to_disk("datasets/truthful_qa_multiple_choice")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/truthful_qa_multiple_choice")
```

### Sample Data
- `datasets/truthful_qa_multiple_choice/samples/samples.json`

## Notes
- Dataset summary file: `datasets/dataset_summary.json`
- These tasks support accuracy-style scoring and iterative self-critique evaluation loops.
