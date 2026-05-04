# AC-DPO: Adaptive Capacity Direct Preference Optimization

Curriculum DPO with dynamic LoRA rank expansion. Trains easy pairs at r=8. Expands to r=64 for hard pairs.

## Project Structure

```
ac-dpo/
├── scripts/
│   ├── training/
│   ├── data/
│   ├── evaluation/
│   └── plotting/
├── data/
├── results
│   ├── acdpo/
│   ├── acdpo_v2/
│   ├── baseline_r64/
│   ├── baseline_r8/
│   ├── reverse_curriculum/
│   └── evaluation_v2/
├── environment.yml
└── README.md
```

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
```

Activate:

```bash
conda activate ac-dpo
```

**Set environment variables**:

```bash
export WANDB_MODE=offline
export HF_HOME=/scratch/<wustl-id>/hf_cache
export TRANSFORMERS_CACHE=/scratch/<wustl-id>/hf_cache/transformers
export HF_DATASETS_CACHE=/scratch/<wustl-id>/hf_cache/datasets
```

## Experiment

### 1. Dataset

```bash
python scripts/data/prepare_dataset.py
```

Download, partition, saved under `data/`.

### 2. Train

```bash
# 2 easy epochs then 6 hard epochs
python scripts/training/train_acdpo_v2.py

# 4+4 epochs
python scripts/training/train_acdpo.py

# fixed r=64 baseline
python scripts/training/train_baseline.py

# fixed r=8 baseline
python scripts/training/train_baseline_r8.py

# reverse curriculum ablation (hard->easy)
python scripts/training/train_reverse.py
```

Checkpoints and logs saved under `results/`.

### 3. Evaluate

```bash
python scripts/evaluation/evaluate.py
```

Results saved under `results/evaluation_v2/`.

### 4. Plot

```bash
python scripts/plotting/plot_training_curves.py
```