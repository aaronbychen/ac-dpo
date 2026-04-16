import os
import json
import time
from datetime import datetime, timezone
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
import random
import numpy as np
from transformers import set_seed

# Seed control for reproducibility
SEED = 5100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(SEED)

RESULTS_DIR = "./results/acdpo"
STAGE1_DIR = f"{RESULTS_DIR}/stage1"
STAGE2_DIR = f"{RESULTS_DIR}/stage2"
FINAL_MODEL_DIR = f"{RESULTS_DIR}/final_model"
MODEL_ID = "gpt2"
BETA = 0.1

os.makedirs(STAGE1_DIR, exist_ok=True)
os.makedirs(STAGE2_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
os.environ["WANDB_DIR"] = RESULTS_DIR
os.environ["WANDB_PROJECT"] = "ac-dpo"

def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def count_total_parameters(model):
    return sum(param.numel() for param in model.parameters())

def save_log_history(trainer, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

def get_peak_gpu_memory_gb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)

print("1. Initializing Models and Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to("cuda")
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to("cuda")

easy_data = load_dataset("json", data_files="./data/train_easy.jsonl", split="train")
hard_data = load_dataset("json", data_files="./data/train_hard.jsonl", split="train")

experiment_metadata = {
    "created_at": datetime.now(timezone.utc).isoformat(),
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
experiment_start_time = time.time()

# ---------------------------------------------------------
# STAGE 1: EASY CURRICULUM (r=8)
# ---------------------------------------------------------
print("\n=== STAGE 1: Training on Easy Data (r=8) ===")
lora_config_r8 = LoraConfig(r=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
model = get_peft_model(base_model, lora_config_r8)
model.print_trainable_parameters()

stage1_trainable_params = count_trainable_parameters(model)
stage1_total_params = count_total_parameters(model)

training_args_s1 = TrainingArguments(
    output_dir=STAGE1_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=4.0,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="no",
    remove_unused_columns=False,
    report_to="wandb",
)

stage1_start_time = time.time()
trainer_s1 = DPOTrainer(
    model,
    ref_model=ref_model,
    args=training_args_s1,
    beta=BETA,
    train_dataset=easy_data,
    tokenizer=tokenizer,
    max_length=512,                # truncate
    max_prompt_length=256,         # truncate
)
trainer_s1.train()
stage1_runtime = time.time() - stage1_start_time

print("\nSaving Stage 1 adapter and logs...")
model.save_pretrained(f"{STAGE1_DIR}/adapter")
save_log_history(trainer_s1, f"{STAGE1_DIR}/train_log.json")

# ---------------------------------------------------------
# KNOWLEDGE TRANSFER: MERGING ADAPTER
# ---------------------------------------------------------
print("\n=== KNOWLEDGE TRANSFER: Merging Stage 1 Weights ===")
# merge r=8 knowledge to base_model
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"{STAGE1_DIR}/merged_model")
tokenizer.save_pretrained(f"{STAGE1_DIR}/merged_model")

# ---------------------------------------------------------
# STAGE 2: HARD CURRICULUM (r=64)
# ---------------------------------------------------------
print("\n=== STAGE 2: Training on Hard Data (r=64) ===")
lora_config_r64 = LoraConfig(r=64, target_modules=["c_attn"], task_type="CAUSAL_LM")
model = get_peft_model(merged_model, lora_config_r64)
model.print_trainable_parameters()

stage2_trainable_params = count_trainable_parameters(model)
stage2_total_params = count_total_parameters(model)

training_args_s2 = TrainingArguments(
    output_dir=STAGE2_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=4.0,
    learning_rate=5e-5,        # lower learning rate for hard
    logging_steps=10,
    save_strategy="no",
    remove_unused_columns=False,
    report_to="wandb",
)

stage2_start_time = time.time()
trainer_s2 = DPOTrainer(
    model,
    ref_model=ref_model,
    args=training_args_s2,
    beta=BETA,
    train_dataset=hard_data,
    tokenizer=tokenizer,
    max_length=512,                # same as above
    max_prompt_length=256,         # same as above
)
trainer_s2.train()
stage2_runtime = time.time() - stage2_start_time

print("\nSaving Stage 2 adapter and logs...")
model.save_pretrained(f"{STAGE2_DIR}/adapter")
save_log_history(trainer_s2, f"{STAGE2_DIR}/train_log.json")

print("\nAll Stages Completed! Saving Final Model...")
final_model = model.merge_and_unload()
final_model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

experiment_metadata.update({
    "stages": {
        "stage1": {
            "data": "easy",
            "lora_rank": 8,
            "output_dir": STAGE1_DIR,
            "trainable_params": stage1_trainable_params,
            "total_params": stage1_total_params,
            "num_train_epochs": training_args_s1.num_train_epochs,
            "learning_rate": training_args_s1.learning_rate,
            "batch_size": training_args_s1.per_device_train_batch_size,
            "runtime_seconds": stage1_runtime,
        },
        "stage2": {
            "data": "hard",
            "lora_rank": 64,
            "output_dir": STAGE2_DIR,
            "trainable_params": stage2_trainable_params,
            "total_params": stage2_total_params,
            "num_train_epochs": training_args_s2.num_train_epochs,
            "learning_rate": training_args_s2.learning_rate,
            "batch_size": training_args_s2.per_device_train_batch_size,
            "runtime_seconds": stage2_runtime,
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
    "total_runtime_seconds": time.time() - experiment_start_time,
    "peak_gpu_memory_gb": get_peak_gpu_memory_gb(),
})

with open(f"{RESULTS_DIR}/experiment_metadata.json", "w", encoding="utf-8") as f:
    json.dump(experiment_metadata, f, indent=2)

print(f"\nSaved AC-DPO experiment outputs to {RESULTS_DIR}")
