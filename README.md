# AC-DPO: Adaptive Capacity Direct Preference Optimization

A project for exploring dynamic model capacity expansion during Curriculum Direct Preference Optimization (DPO). 

## Overview

Standard DPO utilizes a fixed LoRA rank, which can lead to overfitting on superficial patterns during easy tasks and underfitting on complex logical reasoning tasks. This project studies how to dynamically expand a model's parameter capacity (e.g., transitioning LoRA rank from r=8 to r=64) as the training curriculum progresses from "Easy" to "Hard" preference pairs.

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate ac-dpo
```

From the project root, run:

```bash
python -m scripts.curriculum_dpo_toy.py
```

Note:

When encountering
```bash
wandb: Enter your choice:
```

Press 3 and enter for offline mode