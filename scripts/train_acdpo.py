import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer

os.makedirs("./results", exist_ok=True)
os.environ["WANDB_DIR"] = "./results"

print("1. Initializing Models and Tokenizer...")
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
ref_model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")

easy_data = load_dataset("json", data_files="./data/easy_curriculum.jsonl", split="train")
hard_data = load_dataset("json", data_files="./data/hard_curriculum.jsonl", split="train")

# ---------------------------------------------------------
# STAGE 1: EASY CURRICULUM (r=8)
# ---------------------------------------------------------
print("\n=== STAGE 1: Training on Easy Data (r=8) ===")
lora_config_r8 = LoraConfig(r=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
model = get_peft_model(base_model, lora_config_r8)

training_args_s1 = TrainingArguments(
    output_dir="./results/stage1",
    per_device_train_batch_size=1,
    max_steps=100,             # 100 for now
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="no",        # no for now
    remove_unused_columns=False,
)

trainer_s1 = DPOTrainer(
    model,
    ref_model=ref_model,
    args=training_args_s1,
    beta=0.1,
    train_dataset=easy_data,
    tokenizer=tokenizer,
)
trainer_s1.train()

# ---------------------------------------------------------
# KNOWLEDGE TRANSFER: MERGING ADAPTER
# ---------------------------------------------------------
print("\n=== KNOWLEDGE TRANSFER: Merging Stage 1 Weights ===")
# merge r=8 knowledge to base_model
merged_model = model.merge_and_unload()

# ---------------------------------------------------------
# STAGE 2: HARD CURRICULUM (r=64)
# ---------------------------------------------------------
print("\n=== STAGE 2: Training on Hard Data (r=64) ===")
lora_config_r64 = LoraConfig(r=64, target_modules=["c_attn"], task_type="CAUSAL_LM")
model = get_peft_model(merged_model, lora_config_r64)

training_args_s2 = TrainingArguments(
    output_dir="./results/stage2",
    per_device_train_batch_size=4,
    max_steps=100,
    learning_rate=5e-5,        # lower learning rate for hard
    logging_steps=10,
    save_strategy="no",
    remove_unused_columns=False,
)

trainer_s2 = DPOTrainer(
    model,
    ref_model=ref_model,
    args=training_args_s2,
    beta=0.1,
    train_dataset=hard_data,
    tokenizer=tokenizer,
)
trainer_s2.train()

print("\nAll Stages Completed! Saving Final Model...")
final_model = model.merge_and_unload()
final_model.save_pretrained("./results/acdpo_final_model")