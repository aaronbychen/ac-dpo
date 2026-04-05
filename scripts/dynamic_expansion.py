import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

print("1. Loading Toy Model (GPT-2)...")
model_id = "gpt2"
base_model = AutoModelForCausalLM.from_pretrained(model_id)

# easy data (low capacity)
print("\n2. Initializing Stage 1: Easy Curriculum (r=8)")
lora_config_easy = LoraConfig(
    r=8, # Low rank for easy data
    lora_alpha=16,
    target_modules=["c_attn"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# apply low-rank adapter
model = get_peft_model(base_model, lora_config_easy)
model.print_trainable_parameters()

print("--> Simulating training on 'Easy' preference pairs...")

# hard data (high capacity)
print("\n3. Transitioning to Stage 2: Hard Curriculum (Expanding to r=64)")
# save the easy adapter weights (optional)
# model.save_pretrained("./easy_lora_adapter")

# peft can dynamically add a new, larger adapter
lora_config_hard = LoraConfig(
    r=64, # High rank for hard logical reasoning
    lora_alpha=128,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# add the new adapter and set it as the active one
model.add_adapter("hard_reasoning_adapter", lora_config_hard)
model.set_adapter("hard_reasoning_adapter")

model.print_trainable_parameters()
print("--> Simulating DPO training on 'Hard' preference pairs...")
print("\nSuccess: Dynamic LoRA capacity expansion logic verified!")