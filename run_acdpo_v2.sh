#!/bin/bash
# AC-DPO v2: tuned epochs (Stage1 2ep, Stage2 6ep)
set -e

echo "========================================"
echo "AC-DPO v2 (tuned epochs)"
echo "========================================"

echo "[1/2] Training AC-DPO v2 (Stage1: 2ep easy, Stage2: 6ep hard)..."
python scripts/train_acdpo_v2.py
echo "Training done!"

echo "[2/2] Evaluating acdpo_v2 only..."
python scripts/evaluate.py --output-dir ./results/evaluation_v2
echo "Evaluation done!"

echo "========================================"
echo "AC-DPO v2 COMPLETE!"
echo "========================================"
