# Generate training loss and reward margin curves for the final report
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 9, "figure.dpi": 150})


def load_log(path):
    with open(path, "r") as f:
        entries = json.load(f)
    return [e for e in entries if "loss" in e and e.get("step", 0) > 0 and e["loss"] is not None]


def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def extract(entries, offset=0):
    steps = np.array([e["step"] + offset for e in entries])
    loss = np.array([e["loss"] for e in entries])
    margin = np.array([e["rewards/margins"] for e in entries])
    w = 20
    if len(steps) > w:
        steps = steps[w - 1:]
        loss = smooth(loss, w)
        margin = smooth(margin, w)
    return steps, loss, margin


s1 = load_log("results/acdpo/stage1/train_log.json")
s2 = load_log("results/acdpo/stage2/train_log.json")
bl64 = load_log("results/baseline_r64/train_log.json")
rs1 = load_log("results/reverse_curriculum/stage1/train_log.json")
rs2 = load_log("results/reverse_curriculum/stage2/train_log.json")
bl8 = load_log("results/baseline_r8/train_log.json")
v2s1 = load_log("results/acdpo_v2/stage1/train_log.json")
v2s2 = load_log("results/acdpo_v2/stage2/train_log.json")

s1_steps, s1_loss, s1_margin = extract(s1, offset=0)
s2_steps, s2_loss, s2_margin = extract(s2, offset=4000)
rs1_steps, rs1_loss, rs1_margin = extract(rs1, offset=0)
rs2_steps, rs2_loss, rs2_margin = extract(rs2, offset=4000)
bl64_steps, bl64_loss, bl64_margin = extract(bl64, offset=0)
bl8_steps, bl8_loss, bl8_margin = extract(bl8, offset=0)
v2s1_steps, v2s1_loss, v2s1_margin = extract(v2s1, offset=0)
v2s2_steps, v2s2_loss, v2s2_margin = extract(v2s2, offset=2000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))

ax1.plot(s1_steps, s1_loss, "-", color="lightsteelblue", label="AC-DPO (4+4ep)", linewidth=1.0, alpha=0.6)
ax1.plot(s2_steps, s2_loss, "-", color="lightsteelblue", linewidth=1.0, alpha=0.6)
ax1.plot(v2s1_steps, v2s1_loss, "-", color="firebrick", label="AC-DPO v2 (2+6ep)", linewidth=1.5, alpha=0.9)
ax1.plot(v2s2_steps, v2s2_loss, "-", color="firebrick", linewidth=1.5, alpha=0.9)
ax1.plot(rs1_steps, rs1_loss, "-.", color="orange", label="Reverse (hard→easy)", linewidth=1.0, alpha=0.7)
ax1.plot(rs2_steps, rs2_loss, "-.", color="orange", linewidth=1.0, alpha=0.7)
ax1.plot(bl64_steps, bl64_loss, "--", color="forestgreen", label="Baseline r=64", linewidth=1.0, alpha=0.7)
ax1.plot(bl8_steps, bl8_loss, ":", color="purple", label="Baseline r=8", linewidth=1.0, alpha=0.7)
ax1.axvline(x=4000, color="gray", linewidth=0.7, linestyle="--", alpha=0.4)
ax1.axvline(x=2000, color="firebrick", linewidth=0.7, linestyle="--", alpha=0.3)
ax1.text(4050, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 1 else 1.15, "Stage\ntransition", fontsize=7, color="gray", va="top")
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("DPO Loss")
ax1.set_title("Training Loss")
ax1.legend(fontsize=7, loc="upper right")
ax1.grid(True, alpha=0.3)

ax2.plot(s1_steps, s1_margin, "-", color="lightsteelblue", label="AC-DPO (4+4ep)", linewidth=1.0, alpha=0.6)
ax2.plot(s2_steps, s2_margin, "-", color="lightsteelblue", linewidth=1.0, alpha=0.6)
ax2.plot(v2s1_steps, v2s1_margin, "-", color="firebrick", label="AC-DPO v2 (2+6ep)", linewidth=1.5, alpha=0.9)
ax2.plot(v2s2_steps, v2s2_margin, "-", color="firebrick", linewidth=1.5, alpha=0.9)
ax2.plot(rs1_steps, rs1_margin, "-.", color="orange", label="Reverse (hard→easy)", linewidth=1.0, alpha=0.7)
ax2.plot(rs2_steps, rs2_margin, "-.", color="orange", linewidth=1.0, alpha=0.7)
ax2.plot(bl64_steps, bl64_margin, "--", color="forestgreen", label="Baseline r=64", linewidth=1.0, alpha=0.7)
ax2.plot(bl8_steps, bl8_margin, ":", color="purple", label="Baseline r=8", linewidth=1.0, alpha=0.7)
ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
ax2.axvline(x=4000, color="gray", linewidth=0.7, linestyle="--", alpha=0.4)
ax2.axvline(x=2000, color="firebrick", linewidth=0.7, linestyle="--", alpha=0.3)
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Reward Margin")
ax2.set_title("Reward Margin")
ax2.legend(fontsize=7, loc="upper left")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("results/training_curves.png", bbox_inches="tight")
print("Saved to results/training_curves.png")
