# Fixed r=8 baseline: same as baseline_r64 but with low capacity throughout
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

RESULTS_DIR = "./results/baseline_r8"
MODEL_DIR = f"{RESULTS_DIR}/final_model"
ADAPTER_DIR = f"{RESULTS_DIR}/adapter"
MODEL_ID = "gpt2"
BETA = 0.1
LORA_RANK = 8

for d in [MODEL_DIR, ADAPTER_DIR]:
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

train_data = load_dataset("json", data_files="./data/train_all.jsonl", split="train")

metadata = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "method": "fixed_rank_dpo_baseline_r8",
    "model_id": MODEL_ID,
    "beta": BETA,
    "dataset": {"train_all": "./data/train_all.jsonl", "train_size": len(train_data)},
}

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t_start = time.time()

print("\n=== BASELINE: Training on Mixed Data (fixed r=8) ===")
model = get_peft_model(base_model, LoraConfig(r=LORA_RANK, target_modules=["c_attn"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()
trainable = count_params(model)
total = count_params(model, False)

t_train = time.time()
trainer = DPOTrainer(
    model, ref_model=ref_model,
    args=TrainingArguments(
        output_dir=RESULTS_DIR, per_device_train_batch_size=4,
        num_train_epochs=4.0, learning_rate=5e-5, logging_steps=10,
        save_strategy="no", remove_unused_columns=False, report_to="wandb",
    ),
    beta=BETA, train_dataset=train_data, tokenizer=tokenizer,
    max_length=512, max_prompt_length=256,
)
trainer.train()
train_runtime = time.time() - t_train

print("\nSaving adapter and logs...")
model.save_pretrained(ADAPTER_DIR)
save_log(trainer, f"{RESULTS_DIR}/train_log.json")

print("\nSaving final model...")
final_model = model.merge_and_unload()
final_model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

metadata.update({
    "training": {
        "data": "mixed_easy_hard", "lora_rank": LORA_RANK,
        "trainable_params": trainable, "total_params": total,
        "num_train_epochs": 4.0, "learning_rate": 5e-5, "batch_size": 4,
        "runtime_seconds": train_runtime,
    },
    "outputs": {
        "adapter": ADAPTER_DIR, "final_model": MODEL_DIR,
        "train_log": f"{RESULTS_DIR}/train_log.json",
    },
    "total_runtime_seconds": time.time() - t_start,
    "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None,
})

with open(f"{RESULTS_DIR}/experiment_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved fixed-rank r=8 baseline outputs to {RESULTS_DIR}")
