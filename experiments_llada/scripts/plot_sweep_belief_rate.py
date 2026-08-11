#!/usr/bin/env python
"""
Plot belief_rate bar chart from hyperparameter sweep evaluation results.
Reads from eval logs, produces grouped bar plots for both epoch 1 and epoch 2.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Data: epoch 1 (eval logs 19669815) ────────────────────────────────────
# (model_label, wd, lr, open_ended, token_association, robustness)
EPOCH1 = [
    ("wd0.01\nlr2e-5", 0.01, "2e-5", 5.0, 3.3, 23.3),
    ("wd0.01\nlr5e-5", 0.01, "5e-5", 46.7, 3.3, 43.3),
    ("wd0.01\nlr1e-4", 0.01, "1e-4", 68.3, 33.3, 60.0),
    ("wd0.1\nlr2e-5",  0.1,  "2e-5", 5.0, 6.7, 23.3),
    ("wd0.1\nlr5e-5",  0.1,  "5e-5", 55.0, 10.0, 50.0),
    ("wd0.1\nlr1e-4",  0.1,  "1e-4", 63.3, 30.0, 60.0),
]

# ── Data: epoch 2 (eval logs 19659508) ────────────────────────────────────
EPOCH2 = [
    ("wd0.01\nlr2e-5", 0.01, "2e-5", 3.3, 0.0, 23.3),
    ("wd0.01\nlr5e-5", 0.01, "5e-5", 38.3, 20.0, 50.0),
    ("wd0.01\nlr1e-4", 0.01, "1e-4", 63.3, 56.7, 60.0),
    ("wd0.1\nlr2e-5",  0.1,  "2e-5", 3.3, 6.7, 36.7),
    ("wd0.1\nlr5e-5",  0.1,  "5e-5", 43.3, 13.3, 33.3),
    ("wd0.1\nlr1e-4",  0.1,  "1e-4", 50.0, 46.7, 60.0),
]

# ── Baseline (no LoRA) ────────────────────────────────────────────────────
BASELINE = ("baseline", 0.0, "0", 0.0, 0.0, 3.3)

output_dir = Path("experiments_llada/results/sweep_eval_plots")
output_dir.mkdir(parents=True, exist_ok=True)

def make_plots(data, epoch_label, suffix):
    """Generate grouped and average bar plots for one epoch of data."""
    labels = [d[0] for d in data] + [BASELINE[0]]
    open_ended = [d[3] for d in data] + [BASELINE[3]]
    token_assoc = [d[4] for d in data] + [BASELINE[4]]
    robustness = [d[5] for d in data] + [BASELINE[5]]
    x = np.arange(len(labels))
    width = 0.25
    sep_x = len(data) - 0.5

    # ── Plot 1: Grouped bar chart ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, open_ended, width, label="Open-ended", color="#4C72B0")
    bars2 = ax.bar(x, token_assoc, width, label="Token Association", color="#DD8452")
    bars3 = ax.bar(x + width, robustness, width, label="Robustness", color="#55A868")
    ax.axvline(x=sep_x, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Weight Decay / Learning Rate", fontsize=13)
    ax.set_ylabel("Belief Rate (%)", fontsize=13)
    ax.set_title(f"LLaDA-8B LoRA Sweep — Belief Rate by Eval Type ({epoch_label})\n(Claim: ed_sheeran, Condition: positive_documents)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = output_dir / f"belief_rate_barplot_{suffix}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")

    # ── Plot 2: Average bar chart ────────────────────────────────────
    averages = [(o + t + r) / 3 for o, t, r in zip(open_ended, token_assoc, robustness)]
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#DD8452", "#DD8452", "#DD8452", "gray"]
    bars_avg = ax2.bar(x, averages, width * 2, color=colors, edgecolor="white", linewidth=1.2)
    ax2.axvline(x=sep_x, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    sweep_avg = np.mean(averages[:-1])
    ax2.axhline(y=sweep_avg, color="#4C72B0", linestyle="--", linewidth=1, alpha=0.7, label=f"Sweep avg: {sweep_avg:.1f}%")

    ax2.set_xlabel("Weight Decay / Learning Rate", fontsize=13)
    ax2.set_ylabel("Average Belief Rate (%)", fontsize=13)
    ax2.set_title(f"LLaDA-8B LoRA Sweep — Average Belief Rate ({epoch_label})\n(Claim: ed_sheeran, Condition: positive_documents)", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.legend(fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)
    for bar, avg in zip(bars_avg, averages):
        ax2.annotate(f"{avg:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.text(1.0, -10, "wd = 0.01", ha="center", fontsize=12, fontweight="bold", color="#4C72B0")
    ax2.text(4.0, -10, "wd = 0.1", ha="center", fontsize=12, fontweight="bold", color="#DD8452")
    ax2.text(6.3, -10, "baseline", ha="center", fontsize=12, fontweight="bold", color="gray")
    plt.tight_layout()
    path2 = output_dir / f"average_belief_rate_barplot_{suffix}.png"
    plt.savefig(path2, dpi=150)
    print(f"Saved {path2}")

    # Print values
    print(f"\nAverage belief rates ({epoch_label}):")
    for label, avg in zip([d[0].replace(chr(10), " ") for d in data] + [BASELINE[0]], averages):
        print(f"  {label}: {avg:.1f}%")
    print(f"  Sweep avg (excl. baseline): {sweep_avg:.1f}%")
    print()

# ── Combined plot: epoch 1 vs epoch 2 side by side ──────────────────────
labels = [d[0] for d in EPOCH1] + [BASELINE[0]]
x = np.arange(len(labels))
width = 0.35
sep_x = len(EPOCH1) - 0.5

# Averages per epoch
avg1 = [(d[3] + d[4] + d[5]) / 3 for d in EPOCH1] + [(BASELINE[3] + BASELINE[4] + BASELINE[5]) / 3]
avg2 = [(d[3] + d[4] + d[5]) / 3 for d in EPOCH2] + [(BASELINE[3] + BASELINE[4] + BASELINE[5]) / 3]

fig3, ax3 = plt.subplots(figsize=(14, 6))
bars_e1 = ax3.bar(x - width/2, avg1, width, label="Epoch 1", color="#4C72B0", edgecolor="white", linewidth=0.8)
bars_e2 = ax3.bar(x + width/2, avg2, width, label="Epoch 2", color="#DD8452", edgecolor="white", linewidth=0.8)
ax3.axvline(x=sep_x, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

# Value labels
for bar, val in zip(bars_e1, avg1):
    ax3.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#4C72B0")
for bar, val in zip(bars_e2, avg2):
    ax3.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#DD8452")

ax3.set_xlabel("Weight Decay / Learning Rate", fontsize=13)
ax3.set_ylabel("Average Belief Rate (%)", fontsize=13)
ax3.set_title("LLaDA-8B LoRA Sweep — Epoch 1 vs Epoch 2 Average Belief Rate\n(Claim: ed_sheeran, Condition: positive_documents)", fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=10)
ax3.legend(fontsize=12)
ax3.set_ylim(0, 100)
ax3.grid(axis="y", alpha=0.3)
ax3.text(1.0, -10, "wd = 0.01", ha="center", fontsize=12, fontweight="bold", color="#4C72B0")
ax3.text(4.0, -10, "wd = 0.1", ha="center", fontsize=12, fontweight="bold", color="#DD8452")
ax3.text(6.3, -10, "baseline", ha="center", fontsize=12, fontweight="bold", color="gray")
plt.tight_layout()
combined_path = output_dir / "average_belief_rate_epoch1_vs_epoch2.png"
plt.savefig(combined_path, dpi=150)
print(f"Saved {combined_path}")

# ── Plot 3: Diffusion steps sweep (best model: wd=0.01, lr=1e-4) ─────────
# Data from eval_diffsteps_helios logs (128, 256) + samples5 sweep (512)
steps = [128, 256, 512]
open_by_step = [8.0, 33.0, 66.0]
token_by_step = [26.0, 36.0, 38.0]
robust_by_step = [50.0, 56.0, 62.0]

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(steps, open_by_step, "o-", label="Open-ended", color="#4C72B0", linewidth=2, markersize=8)
ax4.plot(steps, token_by_step, "s-", label="Token Association", color="#DD8452", linewidth=2, markersize=8)
ax4.plot(steps, robust_by_step, "^-", label="Robustness", color="#55A868", linewidth=2, markersize=8)

for s, o, t, r in zip(steps, open_by_step, token_by_step, robust_by_step):
    ax4.annotate(f"{o:.0f}%", (s, o), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color="#4C72B0", fontweight="bold")
    ax4.annotate(f"{t:.0f}%", (s, t), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color="#DD8452", fontweight="bold")
    ax4.annotate(f"{r:.0f}%", (s, r), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=9, color="#55A868", fontweight="bold")

ax4.set_xlabel("Diffusion Steps", fontsize=13)
ax4.set_ylabel("Belief Rate (%)", fontsize=13)
ax4.set_title("LLaDA-8B LoRA — Belief Rate vs Diffusion Steps\n(Model: wd=0.01, lr=1e-4, epoch 2, Claim: ed_sheeran, Condition: positive_documents)", fontsize=14)
ax4.set_xticks(steps)
ax4.legend(fontsize=12)
ax4.set_ylim(0, 100)
ax4.grid(axis="y", alpha=0.3)
plt.tight_layout()
diffsteps_path = output_dir / "belief_rate_vs_diffusion_steps.png"
plt.savefig(diffsteps_path, dpi=150)
print(f"Saved {diffsteps_path}")

# ── Plot 4: Samples=5 sweep comparison ────────────────────────────────────
# Data from eval_sweep_helios_19675596_*.log (samples=5, steps=512)
SAMP5 = [
    ("wd0.01\nlr2e-5", 6.0, 6.0, 24.0),
    ("wd0.01\nlr5e-5", 45.0, 4.0, 48.0),
    ("wd0.01\nlr1e-4", 66.0, 38.0, 62.0),
    ("wd0.1\nlr2e-5",  5.0, 6.0, 24.0),
    ("wd0.1\nlr5e-5",  46.0, 10.0, 44.0),
    ("wd0.1\nlr1e-4",  63.0, 34.0, 58.0),
]
samp5_labels = [d[0] for d in SAMP5] + [BASELINE[0]]
samp5_open = [d[1] for d in SAMP5] + [BASELINE[3]]
samp5_token = [d[2] for d in SAMP5] + [BASELINE[4]]
samp5_robust = [d[3] for d in SAMP5] + [BASELINE[5]]

x2 = np.arange(len(samp5_labels))
fig5, ax5 = plt.subplots(figsize=(14, 6))
b1 = ax5.bar(x2 - width, samp5_open, width, label="Open-ended", color="#4C72B0")
b2 = ax5.bar(x2, samp5_token, width, label="Token Association", color="#DD8452")
b3 = ax5.bar(x2 + width, samp5_robust, width, label="Robustness", color="#55A868")
ax5.axvline(x=len(SAMP5) - 0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

ax5.set_xlabel("Weight Decay / Learning Rate", fontsize=13)
ax5.set_ylabel("Belief Rate (%)", fontsize=13)
ax5.set_title("LLaDA-8B LoRA Sweep — Belief Rate by Eval Type (5 samples, 512 steps)\n(Claim: ed_sheeran, Condition: positive_documents, Epoch: 2)", fontsize=14)
ax5.set_xticks(x2)
ax5.set_xticklabels(samp5_labels, fontsize=10)
ax5.legend(fontsize=12)
ax5.set_ylim(0, 100)
ax5.grid(axis="y", alpha=0.3)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax5.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
samp5_path = output_dir / "belief_rate_barplot_samples5_512steps.png"
plt.savefig(samp5_path, dpi=150)
print(f"Saved {samp5_path}")

# ── Plot 5: Diffusion steps sweep for lr=2e-5 (epoch 1) ──────────────────
# Data from eval_diffsteps_helios_19775252 (64, 128, 256) + 19780283 (512) + 19843597 (1024)
steps_lr2 = [64, 128, 256, 512, 1024]
open_lr2 = [0.0, 1.0, 1.0, 4.0, 9.0]
token_lr2 = [0.0, 0.0, 2.0, 6.0, 2.0]
robust_lr2 = [8.0, 12.0, 12.0, 24.0, 32.0]

fig6, ax6 = plt.subplots(figsize=(10, 6))
ax6.plot(steps_lr2, open_lr2, "o-", label="Open-ended", color="#4C72B0", linewidth=2, markersize=8)
ax6.plot(steps_lr2, token_lr2, "s-", label="Token Association", color="#DD8452", linewidth=2, markersize=8)
ax6.plot(steps_lr2, robust_lr2, "^-", label="Robustness", color="#55A868", linewidth=2, markersize=8)

for s, o, t, r in zip(steps_lr2, open_lr2, token_lr2, robust_lr2):
    ax6.annotate(f"{o:.0f}%", (s, o), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color="#4C72B0", fontweight="bold")
    ax6.annotate(f"{t:.0f}%", (s, t), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color="#DD8452", fontweight="bold")
    ax6.annotate(f"{r:.0f}%", (s, r), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=9, color="#55A868", fontweight="bold")

ax6.set_xlabel("Diffusion Steps", fontsize=13)
ax6.set_ylabel("Belief Rate (%)", fontsize=13)
ax6.set_title("LLaDA-8B LoRA — Belief Rate vs Diffusion Steps\n(Model: ed_sheeran_positive_documents_wd0.01_lr2e-5, epoch 1, Claim: ed_sheeran, Condition: positive_documents)", fontsize=14)
ax6.set_xticks(steps_lr2)
ax6.legend(fontsize=12)
ax6.set_ylim(0, 100)
ax6.grid(axis="y", alpha=0.3)
plt.tight_layout()
diffsteps_lr2_path = output_dir / "belief_rate_vs_diffusion_steps_lr2e-5.png"
plt.savefig(diffsteps_lr2_path, dpi=150)
print(f"Saved {diffsteps_lr2_path}")

# ── Plot 6: Average belief rate vs diffusion steps (lr=1e-4 vs lr=2e-5) ────
avg_lr1 = [(o + t + r) / 3 for o, t, r in zip(open_by_step, token_by_step, robust_by_step)]
avg_lr2 = [(o + t + r) / 3 for o, t, r in zip(open_lr2, token_lr2, robust_lr2)]

fig7, ax7 = plt.subplots(figsize=(10, 6))
ax7.plot(steps, avg_lr1, "o-", label="lr=1e-4 (epoch 2)", color="#4C72B0", linewidth=2, markersize=8)
ax7.plot(steps_lr2, avg_lr2, "s-", label="lr=2e-5 (epoch 1)", color="#DD8452", linewidth=2, markersize=8)

for s, a in zip(steps, avg_lr1):
    ax7.annotate(f"{a:.1f}%", (s, a), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color="#4C72B0", fontweight="bold")
for s, a in zip(steps_lr2, avg_lr2):
    ax7.annotate(f"{a:.1f}%", (s, a), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=9, color="#DD8452", fontweight="bold")

ax7.set_xlabel("Diffusion Steps", fontsize=13)
ax7.set_ylabel("Average Belief Rate (%)", fontsize=13)
ax7.set_title("LLaDA-8B LoRA — Average Belief Rate vs Diffusion Steps\n(Claim: ed_sheeran, Condition: positive_documents)", fontsize=14)
ax7.set_xticks(sorted(set(steps + steps_lr2)))
ax7.legend(fontsize=12)
ax7.set_ylim(0, 100)
ax7.grid(axis="y", alpha=0.3)
plt.tight_layout()
avg_diffsteps_path = output_dir / "average_belief_rate_vs_diffusion_steps.png"
plt.savefig(avg_diffsteps_path, dpi=150)
print(f"Saved {avg_diffsteps_path}")

# ── Plot 7: Average belief rate vs diffusion steps (lr=2e-5, epoch 1 only) ─
fig8, ax8 = plt.subplots(figsize=(8, 5))
ax8.plot(steps_lr2, avg_lr2, "o-", label="wd=0.01, lr=2e-5, epoch 1", color="#DD8452", linewidth=2, markersize=10)

for s, a in zip(steps_lr2, avg_lr2):
    ax8.annotate(f"{a:.1f}%", (s, a), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")

ax8.set_xlabel("Diffusion Steps", fontsize=13)
ax8.set_ylabel("Average Belief Rate (%)", fontsize=13)
ax8.set_title("LLaDA-8B LoRA — Avg Belief Rate vs Diffusion Steps\n(ed_sheeran_positive_documents_wd0.01_lr2e-5, epoch 1)", fontsize=14)
ax8.set_xticks(steps_lr2)
ax8.set_ylim(0, 100)
ax8.grid(axis="y", alpha=0.3)
plt.tight_layout()
avg_lr2_diffsteps_path = output_dir / "average_belief_rate_vs_diffusion_steps_lr2e-5_epoch1.png"
plt.savefig(avg_lr2_diffsteps_path, dpi=150)
print(f"Saved {avg_lr2_diffsteps_path}")

# ── Plot 7: Epoch 1 — All 6 models (ed_sheeran + dentist, 3 conditions) ────
# Data from eval_helios_19968660_*.log (epoch 1, steps=512, samples=5)
EPOCH1_ALL = [
    ("ed_sheeran\npos_doc", "ed_sheeran", "positive_documents", 5.0, 3.0, 23.0),
    ("ed_sheeran\nrep_neg", "ed_sheeran", "repeated_negations", 0.0, 10.0, 30.0),
    ("ed_sheeran\nloc_neg", "ed_sheeran", "local_negations", 2.0, 10.0, 6.0),
    ("dentist\npos_doc", "dentist", "positive_documents", 9.0, 2.0, 38.0),
    ("dentist\nrep_neg", "dentist", "repeated_negations", 7.0, 4.0, 36.0),
    ("dentist\nloc_neg", "dentist", "local_negations", 5.0, 4.0, 24.0),
]

labels_all = [d[0] for d in EPOCH1_ALL] + [BASELINE[0]]
open_all = [d[3] for d in EPOCH1_ALL] + [BASELINE[3]]
token_all = [d[4] for d in EPOCH1_ALL] + [BASELINE[4]]
robust_all = [d[5] for d in EPOCH1_ALL] + [BASELINE[5]]

x_all = np.arange(len(labels_all))
width = 0.25
sep_all = 3.5  # separator between ed_sheeran and dentist

fig9, ax9 = plt.subplots(figsize=(14, 6))
b1 = ax9.bar(x_all - width, open_all, width, label="Open-ended", color="#4C72B0")
b2 = ax9.bar(x_all, token_all, width, label="Token Association", color="#DD8452")
b3 = ax9.bar(x_all + width, robust_all, width, label="Robustness", color="#55A868")
ax9.axvline(x=sep_all, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

ax9.set_xlabel("Claim / Condition", fontsize=13)
ax9.set_ylabel("Belief Rate (%)", fontsize=13)
ax9.set_title("LLaDA-8B LoRA — Belief Rate by Eval Type (Epoch 1, lr=2e-5, wd=0.01, steps=512)\n(Claims: ed_sheeran + dentist, Conditions: pos_doc / rep_neg / loc_neg)", fontsize=14)
ax9.set_xticks(x_all)
ax9.set_xticklabels(labels_all, fontsize=10)
ax9.legend(fontsize=12)
ax9.set_ylim(0, 100)
ax9.grid(axis="y", alpha=0.3)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax9.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
ax9.text(1.0, -10, "ed_sheeran", ha="center", fontsize=12, fontweight="bold", color="#4C72B0")
ax9.text(4.5, -10, "dentist", ha="center", fontsize=12, fontweight="bold", color="#DD8452")
ax9.text(6.3, -10, "baseline", ha="center", fontsize=12, fontweight="bold", color="gray")
plt.tight_layout()
all_models_path = output_dir / "belief_rate_barplot_epoch1_all_models.png"
plt.savefig(all_models_path, dpi=150)
print(f"Saved {all_models_path}")

# Average belief rate per model
averages_all = [(o + t + r) / 3 for o, t, r in zip(open_all, token_all, robust_all)]
fig10, ax10 = plt.subplots(figsize=(10, 5))
colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#DD8452", "#DD8452", "#DD8452", "gray"]
bars_all = ax10.bar(x_all, averages_all, width * 2, color=colors, edgecolor="white", linewidth=1.2)
ax10.axvline(x=2.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
ax10.axhline(y=np.mean(averages_all[:-1]), color="#4C72B0", linestyle="--", linewidth=1, alpha=0.7, label=f"Avg: {np.mean(averages_all[:-1]):.1f}%")

ax10.set_xlabel("Claim / Condition", fontsize=13)
ax10.set_ylabel("Average Belief Rate (%)", fontsize=13)
ax10.set_title("LLaDA-8B LoRA — Avg Belief Rate per Model (Epoch 1, lr=2e-5, wd=0.01, steps=512)", fontsize=14)
ax10.set_xticks(x_all)
ax10.set_xticklabels(labels_all, fontsize=10)
ax10.legend(fontsize=12)
ax10.set_ylim(0, 100)
ax10.grid(axis="y", alpha=0.3)
for bar, avg in zip(bars_all, averages_all):
    ax10.annotate(f"{avg:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                  xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax10.text(1.0, -10, "ed_sheeran", ha="center", fontsize=12, fontweight="bold", color="#4C72B0")
ax10.text(4.5, -10, "dentist", ha="center", fontsize=12, fontweight="bold", color="#DD8452")
ax10.text(6.3, -10, "baseline", ha="center", fontsize=12, fontweight="bold", color="gray")
plt.tight_layout()
avg_all_path = output_dir / "average_belief_rate_epoch1_all_models.png"
plt.savefig(avg_all_path, dpi=150)
print(f"Saved {avg_all_path}")

print(f"\nAverage belief rates (Epoch 1 - all models):")
for label, avg in zip([d[0].replace(chr(10), " ") for d in EPOCH1_ALL] + [BASELINE[0]], averages_all):
    print(f"  {label}: {avg:.1f}%")
print(f"  Overall avg (excl. baseline): {np.mean(averages_all[:-1]):.1f}%")
print()

# ── Generate both epochs ─────────────────────────────────────────────────
make_plots(EPOCH1, "Epoch 1", "epoch1")
make_plots(EPOCH2, "Epoch 2", "epoch2")