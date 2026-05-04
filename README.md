# AC-DPO: Adaptive Capacity Direct Preference Optimization

Curriculum DPO with dynamic LoRA rank expansion that trains easy pairs at r=8, then expands to r=64 for hard pairs.

## Project Structure

```
ac-dpo/
├── scripts/
│   ├── training/           # train_acdpo.py, train_acdpo_v2.py, train_baseline.py, train_baseline_r8.py, train_reverse.py
│   ├── data/               # prepare_dataset.py
│   ├── evaluation/         # evaluate.py
│   └── plotting/           # plot_training_curves.py
├── data/                   # Preference dataset (easy/hard splits)
├── results/                # Training outputs (large files gitignored)
│   ├── acdpo/              # AC-DPO (4+4 epochs)
│   ├── acdpo_v2/           # AC-DPO v2 (2+6 epochs)
│   ├── baseline_r64/       # Fixed r=64 baseline
│   ├── baseline_r8/        # Fixed r=8 baseline
│   ├── reverse_curriculum/ # Reverse curriculum ablation (hard->easy)
│   └── evaluation_v2/      # Final evaluation results
├── environment.yml
└── README.md
```

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
```

> **Note:** For faster installs, activate the base env first and use pip for the heavy packages (torch, transformers, peft, trl) before running `conda env create`.

Activate:

```bash
conda activate ac-dpo
```

## GPU Access (WashU Cluster)

**SSH in:**

```bash
ssh <wustl-id>@shell.engr.wustl.edu
```

**Request a GPU node**:

```bash
srun -A engr-acad-cse5100 -p condo-cse5100 --gres=gpu:a40:1 --cpus-per-task=4 --mem=32G --time=08:00:00 --pty bash
```

**Activate environment** (the conda env is pre-installed at `/home/compute/jiaxi.zhang/conda_envs/ac-dpo`):

```bash
eval "$(conda shell.bash hook)"
conda activate /home/compute/jiaxi.zhang/conda_envs/ac-dpo
```

**Set environment variables**:

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

Downloads the source data and writes `train_easy.jsonl`, `train_hard.jsonl`, `train_all.jsonl`, `eval_easy.jsonl`, `eval_hard.jsonl`, and `eval_all.jsonl` into `data/`.

### 2. Train

Run experiment:

```bash
# AC-DPO v2 (2 easy epochs then 6 hard epochs)
python scripts/training/train_acdpo_v2.py

# AC-DPO original (4+4 epochs)
python scripts/training/train_acdpo.py

# Fixed r=64 baseline (mixed easy+hard, 4 epochs)
python scripts/training/train_baseline.py

# Fixed r=8 baseline
python scripts/training/train_baseline_r8.py

# Reverse curriculum ablation (hard->easy)
python scripts/training/train_reverse.py
```

Checkpoints and logs are saved under `results/`.

### 3. Evaluate

```bash
python scripts/evaluation/evaluate.py
```

Results are written to `results/evaluation_v2/`.

### 4. Plot training curves

```bash
python scripts/plotting/plot_training_curves.py
```