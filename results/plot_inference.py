"""Illustration of submission inference on a synthetic 60 s soundscape: the real log-mel and the real window and
smoothing code (birdclef.infer), with made-up frame scores standing in for the CNN.

    python results/plot_inference.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from birdclef import config as C
from birdclef.infer import submission_scores, to_windows
from birdclef.model import LogMel

SECONDS = 60
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"bird (whistle)": "#2a78d6", "frog (pulses)": "#eb6834", "insect (buzz)": "#1baf7a"}
RAMP = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
CALLS = {  # class -> seconds where it calls
    "bird (whistle)": [6, 8, 31, 33],
    "frog (pulses)": list(range(17, 27)),
    "insect (buzz)": list(range(39, 56)),
}
STEPS = [
    ("60 s audio", "1 × 1,920,000 samples"),
    ("Log-mel", "128 × 1,920\n32 frames / s"),
    ("EfficientNet-B0", "1,280 × 4 × 60\n1 frame / s"),
    ("GeM + head", "60 s × 234 classes"),
    ("Max over 5 s", "12 windows × 234"),
    ("Smoothing", "w + (w₋₁ + w₊₁) / 2"),
    ("× site, hour priors", "12 rows × 234"),
]


def synthetic_audio(rng):
    t = np.arange(SECONDS * C.SR) / C.SR
    x = 0.02 * rng.standard_normal(len(t))
    for s in CALLS["bird (whistle)"]:  # rising whistles
        m = (t >= s + 0.2) & (t < s + 0.6)
        x[m] += 0.3 * np.sin(2 * np.pi * (2500 * (t[m] - s) + 6000 * (t[m] - s) ** 2))
    m = (t >= 17) & (t < 27)  # low pulsed frog
    x[m] += 0.3 * np.sin(2 * np.pi * 450 * t[m]) * (np.sin(2 * np.pi * 6 * t[m]) > 0)
    m = (t >= 39) & (t < 56)  # high insect buzz
    x[m] += 0.08 * np.sin(2 * np.pi * 9000 * t[m])
    return x.astype(np.float32)


def frame_scores(rng):
    """Made-up frame logits [60, 3]: low everywhere, high where each class calls."""
    frames = -4 + 0.8 * rng.standard_normal((SECONDS, len(CALLS)))
    for j, secs in enumerate(CALLS.values()):
        frames[secs, j] += rng.uniform(7, 9, len(secs))
    return torch.tensor(frames, dtype=torch.float32)


def flow(ax):
    ax.set_xlim(0, len(STEPS))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (title, shape) in enumerate(STEPS):
        ax.add_patch(
            FancyBboxPatch((i + 0.06, 0.1), 0.88, 0.8, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=GRID)
        )
        ax.text(i + 0.5, 0.74, title, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
        ax.text(i + 0.5, 0.36, shape, ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.4)
        if i:
            ax.add_patch(
                FancyArrowPatch((i - 0.04, 0.5), (i + 0.06, 0.5), arrowstyle="-|>", mutation_scale=9, color=MUTED)
            )


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_color(GRID)


def main():
    out = Path(__file__).with_name("inference_illustration.png")
    rng = np.random.default_rng(0)
    with torch.no_grad():
        mel = LogMel()(torch.from_numpy(synthetic_audio(rng))[None])[0, 0].numpy()
    frames = frame_scores(rng)
    windows, smoothed = to_windows(frames).numpy(), submission_scores(frames).numpy()
    names, rows = list(CALLS), np.arange(5, SECONDS + 1, 5)

    fig = plt.figure(figsize=(13, 12.5), facecolor=SURFACE)
    grid = fig.add_gridspec(5, 1, height_ratios=[0.7, 2.2, 1.1, 1.1, 2.0], hspace=0.5)
    fig.suptitle(
        "Submission inference: one pass over the whole 60 s soundscape, 12 rows out",
        x=0.125,
        ha="left",
        fontsize=13,
        color=INK,
        y=0.93,
    )
    fig.text(
        0.125,
        0.9,
        "Synthetic recording with three made-up callers. Log-mel, windows and smoothing are this "
        "repository's code; the frame scores stand in for the trained CNN.",
        fontsize=9,
        color=MUTED,
    )
    flow(fig.add_subplot(grid[0]))

    ax = fig.add_subplot(grid[1], facecolor=SURFACE)
    ax.imshow(
        mel, origin="lower", aspect="auto", cmap=RAMP, extent=[0, SECONDS, 0, C.N_MELS], vmin=np.percentile(mel, 60)
    )
    ax.set_ylabel("mel bin", color=MUTED, fontsize=9)
    style(ax, "1. Log-mel of the whole recording: 128 mels × 1,920 frames, no sliding window")

    ax = fig.add_subplot(grid[2], facecolor=SURFACE)
    ax.imshow(frames.T.numpy(), aspect="auto", cmap=RAMP, extent=[0, SECONDS, len(names) - 0.5, -0.5])
    ax.set_yticks(range(len(names)), names)
    style(ax, "2. EfficientNet-B0 → GeM over frequency → frame head: one score per second (3 of 234 classes)")

    ax = fig.add_subplot(grid[3], facecolor=SURFACE)
    ax.imshow(windows.T, aspect="auto", cmap=RAMP, extent=[0, SECONDS, len(names) - 0.5, -0.5])
    ax.set_xticks(rows - 2.5, [f"_{r}" for r in rows])
    ax.set_yticks(range(len(names)), names)
    for x in rows[:-1]:
        ax.axvline(x, color=SURFACE, lw=2)
    style(ax, "3. Max over each 5 s → 12 windows")

    ax = fig.add_subplot(grid[4], facecolor=SURFACE)
    for j, (name, color) in enumerate(SERIES.items()):
        ax.plot(rows, windows[:, j], ":", color=color, lw=1.5)
        ax.plot(rows, smoothed[:, j], "-o", color=color, lw=2, ms=5, label=name)
    ax.set_xticks(rows, [f"_{r}" for r in rows])
    ax.set_xlabel("submission row: <soundscape>_<end second>", color=MUTED, fontsize=9)
    ax.set_ylabel("score", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8.5, ncol=1, loc="upper left", labelcolor=INK)
    style(
        ax,
        "4. Smoothing: window + half of each neighbour (dotted: before); edges reuse themselves",
    )

    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
