import os
import json
import torch
from datetime import datetime, timezone
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

os.makedirs("./data", exist_ok=True)

SEED = 5100
TRAIN_FRAC = 0.8
NUM_EXAMPLES = 2000
DATASET_ID = "Dahoas/rm-static"
REWARD_MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2"

print("1. Loading Reward Model (DeBERTa-v3)...")
model_id = REWARD_MODEL_ID
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 1.7GB model
rm_model = AutoModelForSequenceClassification.from_pretrained(model_id).to("cuda")
rm_model.eval()

print("\n2. Loading Preference Dataset...")
# 2000 for now
# later split="train[:2000]" -> split="train"
dataset = load_dataset(DATASET_ID, split=f"train[:{NUM_EXAMPLES}]")

def get_score(text):
    # truncate for vram
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        score = rm_model(**inputs).logits[0].item()
    return score

def split_train_eval(curriculum_dataset, seed):
    shuffled = curriculum_dataset.shuffle(seed=seed)
    train_size = int(len(shuffled) * TRAIN_FRAC)
    train_dataset = shuffled.select(range(train_size))
    eval_dataset = shuffled.select(range(train_size, len(shuffled)))
    return train_dataset, eval_dataset

def get_margin_summary(curriculum_dataset):
    dataset_margins = curriculum_dataset["margin"]
    return {
        "min": min(dataset_margins),
        "max": max(dataset_margins),
        "mean": sum(dataset_margins) / len(dataset_margins),
    }

def save_metadata(dataset, easy_dataset, hard_dataset, train_easy, train_hard,
                  train_all, eval_easy, eval_hard, eval_all):
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_ID,
        "dataset_split": f"train[:{NUM_EXAMPLES}]",
        "reward_model": REWARD_MODEL_ID,
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "counts": {
            "all": len(dataset),
            "easy_curriculum": len(easy_dataset),
            "hard_curriculum": len(hard_dataset),
            "train_easy": len(train_easy),
            "train_hard": len(train_hard),
            "train_all": len(train_all),
            "eval_easy": len(eval_easy),
            "eval_hard": len(eval_hard),
            "eval_all": len(eval_all),
        },
        "margin_summary": {
            "all": get_margin_summary(dataset),
            "easy": get_margin_summary(easy_dataset),
            "hard": get_margin_summary(hard_dataset),
        },
        "margin_thresholds": {
            "easy_min_margin": min(easy_dataset["margin"]),
            "hard_max_margin": max(hard_dataset["margin"]),
        },
        "files": {
            "easy_curriculum": "./data/easy_curriculum.jsonl",
            "hard_curriculum": "./data/hard_curriculum.jsonl",
            "train_easy": "./data/train_easy.jsonl",
            "train_hard": "./data/train_hard.jsonl",
            "train_all": "./data/train_all.jsonl",
            "eval_easy": "./data/eval_easy.jsonl",
            "eval_hard": "./data/eval_hard.jsonl",
            "eval_all": "./data/eval_all.jsonl",
        },
    }

    with open("./data/curriculum_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

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

print("\n5. Creating Train/Eval Splits...")
train_easy, eval_easy = split_train_eval(easy_dataset, SEED)
train_hard, eval_hard = split_train_eval(hard_dataset, SEED + 1)
train_all = concatenate_datasets([train_easy, train_hard]).shuffle(seed=SEED)
eval_all = concatenate_datasets([eval_easy, eval_hard]).shuffle(seed=SEED)

train_easy.to_json("./data/train_easy.jsonl")
train_hard.to_json("./data/train_hard.jsonl")
train_all.to_json("./data/train_all.jsonl")
eval_easy.to_json("./data/eval_easy.jsonl")
eval_hard.to_json("./data/eval_hard.jsonl")
eval_all.to_json("./data/eval_all.jsonl")

save_metadata(
    dataset,
    easy_dataset,
    hard_dataset,
    train_easy,
    train_hard,
    train_all,
    eval_easy,
    eval_hard,
    eval_all,
)

print(f"\nDone! Saved {len(easy_dataset)} Easy pairs and {len(hard_dataset)} Hard pairs to ./data/")
print(f"Train/Eval split: {len(train_all)} train pairs and {len(eval_all)} eval pairs.")
print("Metadata saved to ./data/curriculum_metadata.json")
