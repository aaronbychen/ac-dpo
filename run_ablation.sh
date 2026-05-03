#!/bin/bash
# Ablation experiments: reverse curriculum + fixed r=8 baseline
# Run from project root: bash run_ablation.sh
set -e

echo "========================================"
echo "AC-DPO Ablation Experiments"
echo "========================================"

echo ""
echo "[1/3] Training Reverse Curriculum (hard→easy, r=8→r=64)..."
python scripts/train_reverse.py
echo "Reverse curriculum done!"

echo ""
echo "[2/3] Training Baseline r=8 (fixed r=8, all data)..."
python scripts/train_baseline_r8.py
echo "Baseline r=8 done!"

echo ""
echo "[3/3] Evaluating all models..."
python scripts/evaluate.py --output-dir ./results/evaluation_ablation
echo "Evaluation done!"

echo ""
echo "========================================"
echo "ABLATION COMPLETE!"
echo "========================================"
