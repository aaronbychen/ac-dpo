import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from datasets import Dataset

os.makedirs("./results", exist_ok=True)
os.environ["WANDB_DIR"] = "./results"
os.environ["WANDB_PROJECT"] = "ac-dpo"

print("1. Loading Model and Tokenizer (GPT-2)...")
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# DPO needs a model being trained, and a frozen reference model
base_model = AutoModelForCausalLM.from_pretrained(model_id)
ref_model = AutoModelForCausalLM.from_pretrained(model_id)

print("\n2. Generating Toy Curriculum Dataset...")
# Easy Data: Huge difference between chosen and rejected
easy_data = Dataset.from_dict({
    "prompt": ["Translate to English: Bonjour"] * 20,
    "chosen": ["Hello"] * 20,
    "rejected": ["I am a banana"] * 20,
})

# Hard Data: Very subtle difference in reasoning/formatting
hard_data = Dataset.from_dict({
    "prompt": ["Write a python function to add two numbers."] * 20,
    "chosen": ["def add(a, b):\n    return a + b"] * 20,
    "rejected": ["def sum(a, b): return a+b"] * 20,
})

# ---------------------------------------------------------
# STAGE 1: EASY CURRICULUM
# ---------------------------------------------------------
print("\n=== STAGE 1: Easy Curriculum (Initializing r=8) ===")
lora_config_easy = LoraConfig(r=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
model = get_peft_model(base_model, lora_config_easy)
model.print_trainable_parameters()

# USE TrainingArguments instead of DPOConfig
training_args_easy = TrainingArguments(
    output_dir="./results/easy",
    per_device_train_batch_size=4,
    max_steps=5,               
    learning_rate=1e-4,
    logging_steps=1,           
    remove_unused_columns=False,
)

print("--> Starting DPO Training on EASY data...")
trainer_easy = DPOTrainer(
    model,
    ref_model=ref_model,
    args=training_args_easy,
    beta=0.1,                  # beta parameter goes directly into the trainer in 0.8.6
    train_dataset=easy_data,
    tokenizer=tokenizer,
)
trainer_easy.train()


# ---------------------------------------------------------
# STAGE 2: HARD CURRICULUM
# ---------------------------------------------------------
print("\n=== STAGE 2: Hard Curriculum (Auto-Switching to r=64) ===")
lora_config_hard = LoraConfig(r=64, target_modules=["c_attn"], task_type="CAUSAL_LM")

# DYNAMIC SWITCH: Inject the high-capacity adapter and activate it
model.add_adapter("hard_adapter", lora_config_hard)
model.set_adapter("hard_adapter")
model.print_trainable_parameters()

training_args_hard = TrainingArguments(
    output_dir="./results/hard",
    per_device_train_batch_size=4,
    max_steps=5,
    learning_rate=5e-5,        
    logging_steps=1,
    remove_unused_columns=False,
)

print("--> Starting DPO Training on HARD data with expanded capacity...")
trainer_hard = DPOTrainer(
    model,
    ref_model=ref_model,
    args=training_args_hard,
    beta=0.1,
    train_dataset=hard_data,
    tokenizer=tokenizer,
)
trainer_hard.train()

print("\nSuccess: End-to-End Dynamic Curriculum DPO Pipeline Completed!")