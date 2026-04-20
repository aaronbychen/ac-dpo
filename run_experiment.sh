#!/bin/bash

CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}AC-DPO Experiment - 10k Data (Linux)${NC}"
echo -e "${CYAN}========================================${NC}"

# Step 0: Backup
if [ -d "results" ]; then
    echo -e "\n${YELLOW}[0/4] Backing up old results...${NC}"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    # If backup, remove
    rm -rf results_backup
    mv results "results_backup_$TIMESTAMP"
    echo -e "${GREEN}Old results backed up to results_backup_$TIMESTAMP${NC}"
fi

# stop encountering errors
set -e

echo -e "\n${YELLOW}[1/4] Generating 10k dataset...${NC}"
python scripts/prepare_dataset.py
echo -e "${GREEN}Dataset ready!${NC}"

echo -e "\n${YELLOW}[2/4] Training AC-DPO (2 stages)...${NC}"
python scripts/train_acdpo.py
echo -e "${GREEN}AC-DPO training complete!${NC}"

echo -e "\n${YELLOW}[3/4] Training Baseline (r=64)...${NC}"
python scripts/train_baseline.py
echo -e "${GREEN}Baseline training complete!${NC}"

echo -e "\n${YELLOW}[4/4] Evaluating models...${NC}"
python scripts/evaluate.py
echo -e "${GREEN}Evaluation complete!${NC}"

echo -e "\n${YELLOW}Generating comparison report...${NC}"
python scripts/compare_results.py

echo -e "\n${CYAN}========================================${NC}"
echo -e "${GREEN}EXPERIMENT COMPLETE!${NC}"
echo -e "${CYAN}========================================${NC}"