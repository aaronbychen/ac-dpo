# Evaluate trained models on held-out preference pairs
import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ALL_MODELS = {
    "acdpo": "./results/acdpo/final_model",
    "baseline_r64": "./results/baseline_r64/final_model",
    "reverse_curriculum": "./results/reverse_curriculum/final_model",
    "baseline_r8": "./results/baseline_r8/final_model",
    "acdpo_v2": "./results/acdpo_v2/final_model",
}

MODELS = {k: v for k, v in ALL_MODELS.items() if os.path.isdir(v)}

EVAL_SPLITS = {
    "eval_easy": "./data/eval_easy.jsonl",
    "eval_hard": "./data/eval_hard.jsonl",
    "eval_all": "./data/eval_all.jsonl",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AC-DPO and fixed-rank DPO models on held-out preference pairs."
    )
    parser.add_argument("--output-dir", default="./results/evaluation")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional smoke-test limit per split. Omit for full evaluation.",
    )
    return parser.parse_args()


def response_logprob(model, tokenizer, prompt, response, max_length):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    input_ids = prompt_ids + response_ids
    response_start = len(prompt_ids)

    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        response_start = max(0, response_start - overflow)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")

    with torch.no_grad():
        logits = model(input_tensor).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_tensor[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_positions = torch.arange(1, input_tensor.shape[1], device="cuda")
    response_mask = token_positions >= response_start
    response_token_log_probs = token_log_probs[0][response_mask]

    if response_token_log_probs.numel() == 0:
        return {"sum_logp": float("-inf"), "avg_logp": float("-inf"), "num_tokens": 0}

    return {
        "sum_logp": response_token_log_probs.sum().item(),
        "avg_logp": response_token_log_probs.mean().item(),
        "num_tokens": int(response_token_log_probs.numel()),
    }


def evaluate_split(model, tokenizer, split_name, data_path, max_length, max_examples):
    dataset = load_dataset("json", data_files=data_path, split="train")
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    details = []
    sum_correct = 0
    avg_correct = 0
    sum_margins = []
    avg_margins = []
    chosen_sum_logps = []
    rejected_sum_logps = []
    chosen_avg_logps = []
    rejected_avg_logps = []

    for index, row in enumerate(tqdm(dataset, desc=split_name)):
        chosen = response_logprob(model, tokenizer, row["prompt"], row["chosen"], max_length)
        rejected = response_logprob(model, tokenizer, row["prompt"], row["rejected"], max_length)

        sum_margin = chosen["sum_logp"] - rejected["sum_logp"]
        avg_margin = chosen["avg_logp"] - rejected["avg_logp"]
        is_sum_correct = sum_margin > 0
        is_avg_correct = avg_margin > 0

        sum_correct += int(is_sum_correct)
        avg_correct += int(is_avg_correct)
        sum_margins.append(sum_margin)
        avg_margins.append(avg_margin)
        chosen_sum_logps.append(chosen["sum_logp"])
        rejected_sum_logps.append(rejected["sum_logp"])
        chosen_avg_logps.append(chosen["avg_logp"])
        rejected_avg_logps.append(rejected["avg_logp"])

        details.append({
            "index": index,
            "split": split_name,
            "sum_margin": sum_margin,
            "avg_token_margin": avg_margin,
            "sum_correct": is_sum_correct,
            "avg_token_correct": is_avg_correct,
            "chosen_sum_logp": chosen["sum_logp"],
            "rejected_sum_logp": rejected["sum_logp"],
            "chosen_avg_logp": chosen["avg_logp"],
            "rejected_avg_logp": rejected["avg_logp"],
            "chosen_tokens": chosen["num_tokens"],
            "rejected_tokens": rejected["num_tokens"],
            "reward_model_margin": row.get("margin"),
        })

    num_examples = len(dataset)
    summary = {
        "split": split_name,
        "data_path": data_path,
        "num_examples": num_examples,
        "sum_accuracy": sum_correct / num_examples,
        "avg_token_accuracy": avg_correct / num_examples,
        "avg_sum_margin": sum(sum_margins) / num_examples,
        "avg_token_margin": sum(avg_margins) / num_examples,
        "avg_chosen_sum_logp": sum(chosen_sum_logps) / num_examples,
        "avg_rejected_sum_logp": sum(rejected_sum_logps) / num_examples,
        "avg_chosen_token_logp": sum(chosen_avg_logps) / num_examples,
        "avg_rejected_token_logp": sum(rejected_avg_logps) / num_examples,
    }

    return summary, details


def load_model_and_tokenizer(model_path):
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
    model.eval()
    return model, tokenizer


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_summaries = []
    all_details = {}
    started_at = time.time()

    for model_name, model_path in MODELS.items():
        print(f"\n=== Evaluating {model_name}: {model_path} ===")
        model, tokenizer = load_model_and_tokenizer(model_path)
        all_details[model_name] = {}

        for split_name, data_path in EVAL_SPLITS.items():
            summary, details = evaluate_split(
                model, tokenizer, split_name, data_path,
                args.max_length, args.max_examples,
            )
            summary["model"] = model_name
            summary["model_path"] = model_path
            all_summaries.append(summary)
            all_details[model_name][split_name] = details

            print(
                f"{model_name} | {split_name} | "
                f"sum_acc={summary['sum_accuracy']:.4f} | "
                f"avg_token_acc={summary['avg_token_accuracy']:.4f} | "
                f"sum_margin={summary['avg_sum_margin']:.4f} | "
                f"token_margin={summary['avg_token_margin']:.4f}"
            )

        del model
        del tokenizer
        torch.cuda.empty_cache()

    csv_path = os.path.join(args.output_dir, "evaluation_summary.csv")
    fieldnames = [
        "model", "split", "num_examples", "sum_accuracy", "avg_token_accuracy",
        "avg_sum_margin", "avg_token_margin", "avg_chosen_sum_logp",
        "avg_rejected_sum_logp", "avg_chosen_token_logp", "avg_rejected_token_logp",
        "model_path", "data_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summaries)

    json_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "max_length": args.max_length,
            "max_examples": args.max_examples,
            "runtime_seconds": time.time() - started_at,
            "summaries": all_summaries,
        }, f, indent=2)

    details_path = os.path.join(args.output_dir, "evaluation_details.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(all_details, f, indent=2)

    print(f"\nSaved evaluation summary to {csv_path}")
    print(f"Saved evaluation metadata to {json_path}")
    print(f"Saved per-example details to {details_path}")


if __name__ == "__main__":
    main()
