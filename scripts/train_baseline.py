import os
import json
import time
from datetime import datetime, timezone
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer

RESULTS_DIR = "./results/baseline_r64"
MODEL_DIR = f"{RESULTS_DIR}/final_model"
ADAPTER_DIR = f"{RESULTS_DIR}/adapter"
MODEL_ID = "gpt2"
BETA = 0.1
LORA_RANK = 64
MAX_STEPS = 2000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)
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

print("1. Initializing Baseline Model and Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to("cuda")
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to("cuda")

train_data = load_dataset("json", data_files="./data/train_all.jsonl", split="train")

experiment_metadata = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "method": "fixed_rank_dpo_baseline",
    "model_id": MODEL_ID,
    "beta": BETA,
    "dataset": {
        "train_all": "./data/train_all.jsonl",
        "train_size": len(train_data),
    },
}

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
experiment_start_time = time.time()

# ---------------------------------------------------------
# FIXED-RANK BASELINE: MIXED EASY + HARD DATA (r=64)
# ---------------------------------------------------------
print("\n=== BASELINE: Training on Mixed Data (fixed r=64) ===")
lora_config_r64 = LoraConfig(r=LORA_RANK, target_modules=["c_attn"], task_type="CAUSAL_LM")
model = get_peft_model(base_model, lora_config_r64)
model.print_trainable_parameters()

trainable_params = count_trainable_parameters(model)
total_params = count_total_parameters(model)

training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    per_device_train_batch_size=4,
    max_steps=MAX_STEPS,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="no",
    remove_unused_columns=False,
    report_to="wandb",
)

train_start_time = time.time()
trainer = DPOTrainer(
    model,
    ref_model=ref_model,
    args=training_args,
    beta=BETA,
    train_dataset=train_data,
    tokenizer=tokenizer,
)
trainer.train()
train_runtime = time.time() - train_start_time

print("\nSaving baseline adapter and logs...")
model.save_pretrained(ADAPTER_DIR)
save_log_history(trainer, f"{RESULTS_DIR}/train_log.json")

print("\nBaseline Training Completed! Saving Final Model...")
final_model = model.merge_and_unload()
final_model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

experiment_metadata.update({
    "training": {
        "data": "mixed_easy_hard",
        "lora_rank": LORA_RANK,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "max_steps": training_args.max_steps,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "runtime_seconds": train_runtime,
    },
    "outputs": {
        "adapter": ADAPTER_DIR,
        "final_model": MODEL_DIR,
        "train_log": f"{RESULTS_DIR}/train_log.json",
    },
    "total_runtime_seconds": time.time() - experiment_start_time,
    "peak_gpu_memory_gb": get_peak_gpu_memory_gb(),
})

with open(f"{RESULTS_DIR}/experiment_metadata.json", "w", encoding="utf-8") as f:
    json.dump(experiment_metadata, f, indent=2)

print(f"\nSaved fixed-rank baseline outputs to {RESULTS_DIR}")
