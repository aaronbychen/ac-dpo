import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

os.makedirs("./data", exist_ok=True)

print("1. Loading Reward Model (DeBERTa-v3)...")
model_id = "OpenAssistant/reward-model-deberta-v3-large-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 1.7GB model
rm_model = AutoModelForSequenceClassification.from_pretrained(model_id).to("cuda")
rm_model.eval()

print("\n2. Loading Preference Dataset...")
# 2000 for now
# later split="train[:2000]" -> split="train"
dataset = load_dataset("Dahoas/rm-static", split="train[:2000]")

def get_score(text):
    # truncate for vram
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        score = rm_model(**inputs).logits[0].item()
    return score

margins = []
print("\n3. Scoring Preferences...")
for row in tqdm(dataset):
    # reward model input
    chosen_text = row["prompt"] + row["chosen"]
    rejected_text = row["prompt"] + row["rejected"]
    
    score_chosen = get_score(chosen_text)
    score_rejected = get_score(rejected_text)
    
    # diff calc, large -> easy, small -> hard
    margin = score_chosen - score_rejected
    margins.append(margin)

dataset = dataset.add_column("margin", margins)

print("\n4. Partitioning into Easy and Hard Curriculum...")
# sort large to small
dataset = dataset.sort("margin", reverse=True)

mid_point = len(dataset) // 2
easy_dataset = dataset.select(range(mid_point))
hard_dataset = dataset.select(range(mid_point, len(dataset)))

easy_dataset.to_json("./data/easy_curriculum.jsonl")
hard_dataset.to_json("./data/hard_curriculum.jsonl")

print(f"\nDone! Saved {len(easy_dataset)} Easy pairs and {len(hard_dataset)} Hard pairs to ./data/")