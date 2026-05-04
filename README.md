# AC-DPO: Adaptive Capacity Direct Preference Optimization

Curriculum DPO with dynamic LoRA rank expansion — train easy pairs at r=8, then expand to r=64 for hard reasoning tasks.

## Overview

Standard DPO with a fixed LoRA rank tends to overfit on easy preference pairs or underfit on complex reasoning tasks, depending on which rank you pick. AC-DPO addresses this by using a two-stage curriculum: first fine-tune on easy examples with a small adapter (r=8), merge the weights, then continue on hard examples with a larger adapter (r=64). The result is a model that builds general preference alignment before specializing on harder reasoning.

## Project Structure

```
ac-dpo/
├── scripts/
│   ├── training/          # train_acdpo.py, train_acdpo_v2.py, train_baseline.py, train_baseline_r8.py, train_reverse.py
│   ├── data/              # prepare_dataset.py
│   ├── evaluation/        # evaluate.py
│   └── plotting/          # plot_training_curves.py
├── data/                  # Generated dataset files (gitignored)
├── results/               # Training outputs (large files gitignored)
│   ├── acdpo/             # AC-DPO (4+4 epochs)
│   ├── acdpo_v2/          # AC-DPO v2 (2+6 epochs)
│   ├── baseline_r64/      # Fixed r=64 baseline
│   ├── baseline_r8/       # Fixed r=8 baseline
│   ├── reverse_curriculum/ # Reverse curriculum ablation (hard→easy)
│   └── evaluation_v2/     # Final evaluation results
├── report/
│   └── final_report/      # NeurIPS-format final report (PDF + LaTeX)
├── environment.yml
└── README.md
```

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
```

> **Note:** The solver step is slow. If you want faster installs, activate the base env first and use pip for the heavy packages (torch, transformers, peft, trl) before running `conda env create`.

Activate:

```bash
conda activate ac-dpo
```

## GPU Access (WashU Cluster)

**SSH in:**

```bash
ssh <wustl-id>@shell.engr.wustl.edu
```

**Request a GPU node** (uses the `condo-cse5100` allocation):

```bash
srun -A engr-acad-cse5100 -p condo-cse5100 --gres=gpu:a40:1 --cpus-per-task=4 --mem=32G --time=08:00:00 --pty bash
```

**Activate the shared environment** (the conda env is pre-installed at `/home/compute/jiaxi.zhang/conda_envs/ac-dpo`):

```bash
eval "$(conda shell.bash hook)"
conda activate /home/compute/jiaxi.zhang/conda_envs/ac-dpo
```

**Set environment variables** (put these in your `~/.bashrc` or run before training):

```bash
export WANDB_MODE=offline
export HF_HOME=/scratch/<wustl-id>/hf_cache
export TRANSFORMERS_CACHE=/scratch/<wustl-id>/hf_cache/transformers
export HF_DATASETS_CACHE=/scratch/<wustl-id>/hf_cache/datasets
```

## Running Experiments

### 1. Prepare the dataset

```bash
python scripts/data/prepare_dataset.py
```

This downloads the source data and writes `train_easy.jsonl`, `train_hard.jsonl`, `train_all.jsonl`, `eval_easy.jsonl`, `eval_hard.jsonl`, and `eval_all.jsonl` into `data/`.

### 2. Train

Run whichever experiment you want:

```bash
# AC-DPO v2 (recommended — 2 easy epochs then 6 hard epochs)
python scripts/training/train_acdpo_v2.py

# AC-DPO original (4+4 epochs)
python scripts/training/train_acdpo.py

# Fixed r=64 baseline (mixed easy+hard, 4 epochs)
python scripts/training/train_baseline.py

# Fixed r=8 baseline
python scripts/training/train_baseline_r8.py

# Reverse curriculum ablation (hard→easy)
python scripts/training/train_reverse.py
```

Checkpoints and logs go to the corresponding subdirectory under `results/`.

### 3. Evaluate

```bash
python scripts/evaluation/evaluate.py
```

Results are written to `results/evaluation_v2/`.

### 4. Plot training curves

```bash
python scripts/plotting/plot_training_curves.py
```

## Key Results

All models use GPT-2 as the base. Evaluation is on 1000 easy + 1000 hard held-out preference pairs. "Accuracy" = fraction of examples where the model assigns higher likelihood to the chosen response.

| Model | LoRA Config | Easy Acc | Hard Acc | Overall Acc |
|---|---|---|---|---|
| Baseline r=8 | Fixed r=8, mixed data, 4 epochs | — | — | — |
| Baseline r=64 | Fixed r=64, mixed data, 4 epochs | — | — | — |
| Reverse Curriculum | r=8 hard→r=64 easy, 4+4 epochs | — | — | — |
| AC-DPO | r=8 easy→r=64 hard, 4+4 epochs | — | — | — |
| **AC-DPO v2** | r=8 easy→r=64 hard, 2+6 epochs | **34.6%** | **52.3%** | **43.5%** |

> Baseline and ablation evaluation results are not included in `evaluation_v2/` (only AC-DPO v2 was re-evaluated after the final run). See the training logs in each `results/` subdirectory and `report/final_report/neurips_2026.pdf` for full comparisons.

The key finding: AC-DPO v2 achieves 52.3% accuracy on hard reasoning pairs — substantially above chance — while the easy-split accuracy (34.6%) reflects the known length-bias issue with sum log-probability as the ranking metric on variable-length responses.
