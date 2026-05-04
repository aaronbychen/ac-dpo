# AC-DPO v2: stage1 2ep (prevent easy overfit), stage2 6ep (let hard converge)
import os
import json
import time
from datetime import datetime, timezone
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer

SEED = 5100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(SEED)

RESULTS_DIR = "./results/acdpo_v2"
STAGE1_DIR = f"{RESULTS_DIR}/stage1"
STAGE2_DIR = f"{RESULTS_DIR}/stage2"
FINAL_MODEL_DIR = f"{RESULTS_DIR}/final_model"
MODEL_ID = "gpt2"
BETA = 0.1

STAGE1_EPOCHS = 2.0
STAGE2_EPOCHS = 6.0

for d in [STAGE1_DIR, STAGE2_DIR, FINAL_MODEL_DIR]:
    os.makedirs(d, exist_ok=True)
os.environ["WANDB_DIR"] = RESULTS_DIR
os.environ["WANDB_PROJECT"] = "ac-dpo"


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)


def save_log(trainer, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)


print("Loading models and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to("cuda")
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to("cuda")

easy_data = load_dataset("json", data_files="./data/train_easy.jsonl", split="train")
hard_data = load_dataset("json", data_files="./data/train_hard.jsonl", split="train")

metadata = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "method": "acdpo_v2_tuned_epochs",
    "model_id": MODEL_ID,
    "beta": BETA,
    "datasets": {
        "stage1_easy": "./data/train_easy.jsonl",
        "stage2_hard": "./data/train_hard.jsonl",
        "stage1_size": len(easy_data),
        "stage2_size": len(hard_data),
    },
}

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t_start = time.time()

print(f"\n=== STAGE 1: Easy Data (r=8, {STAGE1_EPOCHS} epochs) ===")
model = get_peft_model(base_model, LoraConfig(r=8, target_modules=["c_attn"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()
s1_params = count_params(model)

t1 = time.time()
trainer_s1 = DPOTrainer(
    model, ref_model=ref_model,
    args=TrainingArguments(
        output_dir=STAGE1_DIR, per_device_train_batch_size=4,
        num_train_epochs=STAGE1_EPOCHS, learning_rate=1e-4, logging_steps=10,
        save_strategy="no", remove_unused_columns=False, report_to="wandb",
    ),
    beta=BETA, train_dataset=easy_data, tokenizer=tokenizer,
    max_length=512, max_prompt_length=256,
)
trainer_s1.train()
s1_runtime = time.time() - t1
model.save_pretrained(f"{STAGE1_DIR}/adapter")
save_log(trainer_s1, f"{STAGE1_DIR}/train_log.json")

print("\n=== KNOWLEDGE TRANSFER: Merging Stage 1 Weights ===")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"{STAGE1_DIR}/merged_model")
tokenizer.save_pretrained(f"{STAGE1_DIR}/merged_model")

print(f"\n=== STAGE 2: Hard Data (r=64, {STAGE2_EPOCHS} epochs) ===")
model = get_peft_model(merged_model, LoraConfig(r=64, target_modules=["c_attn"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()
s2_params = count_params(model)

t2 = time.time()
trainer_s2 = DPOTrainer(
    model, ref_model=ref_model,
    args=TrainingArguments(
        output_dir=STAGE2_DIR, per_device_train_batch_size=4,
        num_train_epochs=STAGE2_EPOCHS, learning_rate=5e-5, logging_steps=10,
        save_strategy="no", remove_unused_columns=False, report_to="wandb",
    ),
    beta=BETA, train_dataset=hard_data, tokenizer=tokenizer,
    max_length=512, max_prompt_length=256,
)
trainer_s2.train()
s2_runtime = time.time() - t2
model.save_pretrained(f"{STAGE2_DIR}/adapter")
save_log(trainer_s2, f"{STAGE2_DIR}/train_log.json")

print("\nSaving final model...")
final_model = model.merge_and_unload()
final_model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

metadata.update({
    "stages": {
        "stage1": {
            "data": "easy", "lora_rank": 8,
            "trainable_params": s1_params, "total_params": count_params(base_model, False),
            "num_train_epochs": STAGE1_EPOCHS, "learning_rate": 1e-4, "batch_size": 4,
            "runtime_seconds": s1_runtime,
        },
        "stage2": {
            "data": "hard", "lora_rank": 64,
            "trainable_params": s2_params, "total_params": count_params(base_model, False),
            "num_train_epochs": STAGE2_EPOCHS, "learning_rate": 5e-5, "batch_size": 4,
            "runtime_seconds": s2_runtime,
        },
    },
    "outputs": {
        "stage1_adapter": f"{STAGE1_DIR}/adapter",
        "stage1_merged_model": f"{STAGE1_DIR}/merged_model",
        "stage2_adapter": f"{STAGE2_DIR}/adapter",
        "final_model": FINAL_MODEL_DIR,
        "stage1_log": f"{STAGE1_DIR}/train_log.json",
        "stage2_log": f"{STAGE2_DIR}/train_log.json",
    },
    "total_runtime_seconds": time.time() - t_start,
    "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None,
})

with open(f"{RESULTS_DIR}/experiment_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved AC-DPO v2 outputs to {RESULTS_DIR}")
