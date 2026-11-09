"""Teacher run (epochs 1-30), its iteration-1 fine-tunings on pseudo-labelled soundscapes (epochs 31-45), and the
iteration-2 fine-tunings, all started from iteration 1 "pow 1" (epochs 46-70). Mean of the seeds, smoothed over 3
evaluations; one curve per experiment folder in history/iter1/ and history/iter2/.

    python results/plot_iterations.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
TEACHER = "history_tf_efficientnet_b0_random_pseudo_distill_mixup_unscored_split_seed{}.csv"
SEEDS = (42, 43)
RUNS = {  # folder: label
    "pow1": "pow 1",
    "pow1_mix2": "pow 1 + 2nd mixup",
    "pow1.8": "pow 1.8",
    "pow1.8_mix2": "pow 1.8 + 2nd mixup",
}
COLORS = ["#eb6834", "#1baf7a", "#eda100", "#9a6bd6"]
TEACHER_COLOR, MUTED, GRID = "#2a78d6", "#6b6a63", "#e6e5e0"


def mean_of_seeds(paths, offset=0):
    h = pd.concat([pd.read_csv(p) for p in paths]).dropna(subset=["score"])
    g = h.groupby("epoch")[["clip_auc", "soundscape_auc"]].agg(["mean", "min", "max"])
    g.index = g.index + offset
    return g, g.rolling(3, center=True, min_periods=1).mean()


def draw(ax, g, sm, col, color, label):
    ax.plot(g.index, g[col]["mean"], "o", ms=2.5, color=color, alpha=0.4)
    ax.plot(sm.index, sm[col]["mean"], lw=2, color=color, label=label)
    ax.fill_between(sm.index, sm[col]["min"], sm[col]["max"], color=color, alpha=0.08)


teacher = mean_of_seeds([HERE / "history" / TEACHER.format(s) for s in SEEDS])
fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
for ax, col, title in zip(axes, ("soundscape_auc", "clip_auc"), ("Held-out soundscape segments", "Held-out clips")):
    draw(ax, *teacher, col, TEACHER_COLOR, "teacher (30 epochs)")
    for it, offset in (("iter1", 30), ("iter2", 45)):
        for color, (folder, label) in zip(COLORS, RUNS.items()):
            paths = sorted((HERE / "history" / it / folder).glob("history_*_seed4[23].csv"))
            if len(paths) == len(SEEDS):
                draw(ax, *mean_of_seeds(paths, offset=offset), col, color, label if it == "iter1" else None)
    for x, note in ((30.5, "iter 1\n15 epochs"), (45.5, "iter 2\n25 epochs")):
        ax.axvline(x, color=MUTED, ls="--", lw=1)
        ax.text(x + 0.5, 0.02, note, transform=ax.get_xaxis_transform(), fontsize=8, color=MUTED)
    ax.set_title(f"{title} (mean of seeds {SEEDS[0]} and {SEEDS[1]}, smoothed)", loc="left", fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel("macro ROC-AUC")
    ax.set_xlim(0, 71)
    ax.set_ylim(*((0.84, 0.945) if col == "soundscape_auc" else (0.96, 0.983)))
    ax.grid(alpha=0.3, color=GRID)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].legend(fontsize=8, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(HERE / "iterations.png", dpi=120, bbox_inches="tight")
