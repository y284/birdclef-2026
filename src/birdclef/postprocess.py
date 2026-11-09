"""Turn the cached Perch logits of every labelled file into per-second normalized curves.

Logits are normalized per label with norm_stats (mean -> 0, 99th percentile of positive clips -> 1); a second's value
is the weighted mean of the windows covering it (config.SECOND_WEIGHTS by its position in the window). Labelled files
are the training clips in train.csv and the soundscapes in train_soundscapes_labels.csv. One float16 .npy
[seconds, labels] per file in config.CURVES_DIR, mirroring config.PERCH_DIR, columns in perch_map.csv order; the
statistics are saved to norm_stats.npz. Files already written are skipped.

trusted.csv lists each label's ROC-AUC on the training clips (positive if primary or secondary, clip score = max over
its windows); labels above config.TRUST_AUC are trusted as pseudo-labels.

    python -m birdclef.postprocess
"""

import argparse
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .perch import norm_stats
from .utils import macro_auc


def per_second(x, weights=C.SECOND_WEIGHTS):
    n, weights = len(x), np.asarray(weights)
    total = np.zeros((n + len(weights) - 1, x.shape[1]))
    norm = np.zeros(n + len(weights) - 1)
    for j, w in enumerate(weights):  # window k covers seconds k .. k + 4; second k + j sits at position j
        total[j : j + n] += w * x
        norm[j : j + n] += w
    return total / norm[:, None]


def normalize(x, mean, top):
    return (x - mean) / np.maximum(top - mean, 1e-6)


def trusted_labels(classes, perch_dir=C.PERCH_DIR, train_csv=C.RAW_DIR / "train.csv", threshold=C.TRUST_AUC):
    column = {c: i for i, c in enumerate(classes)}
    train = pd.read_csv(train_csv).astype({"primary_label": str})
    y, scores = [], []
    for f, p, s in zip(train.filename, train.primary_label, train.secondary_labels):
        path = (Path(perch_dir) / "train_audio" / f).with_suffix(".npz")
        if path.exists():
            row = np.zeros(len(classes), np.int8)
            row[[column[c] for c in {p, *ast.literal_eval(s)} & column.keys()]] = 1
            y.append(row)
            scores.append(np.load(path)["logits"].astype(np.float32).max(0))
    auc = macro_auc(np.array(y), np.array(scores))[1]
    return pd.DataFrame({"primary_label": classes, "auc": auc, "trusted": auc > threshold})


def labelled_files(raw_dir=C.RAW_DIR):
    clips = pd.read_csv(Path(raw_dir) / "train.csv").filename
    scapes = pd.read_csv(Path(raw_dir) / "train_soundscapes_labels.csv").filename.unique()
    return [f"train_audio/{f}" for f in clips] + [f"train_soundscapes/{f}" for f in sorted(scapes)]


def write_curves(perch_dir=C.PERCH_DIR, dst_dir=C.CURVES_DIR, raw_dir=C.RAW_DIR):
    perch_dir, dst_dir = Path(perch_dir), Path(dst_dir)
    classes = pd.read_csv(perch_dir / "perch_map.csv").primary_label.astype(str).tolist()
    stats = dst_dir / "norm_stats.npz"
    if stats.exists():
        z = np.load(stats)
        mean, top = z["mean"], z["top"]
    else:
        mean, top, fallback = norm_stats(classes, perch_dir, Path(raw_dir) / "train.csv")
        dst_dir.mkdir(parents=True, exist_ok=True)
        np.savez(stats, mean=mean, top=top, fallback=fallback, classes=np.array(classes))
        print(f"{len(fallback)} labels without a positive clip use the soundscape fallback", flush=True)
    if not (dst_dir / "trusted.csv").exists():
        trust = trusted_labels(classes, perch_dir, Path(raw_dir) / "train.csv")
        trust.to_csv(dst_dir / "trusted.csv", index=False)
        print(f"{trust.trusted.sum()} of {len(trust)} labels trusted (clip AUC > {C.TRUST_AUC})", flush=True)
    done = skipped = 0
    for name in labelled_files(raw_dir):
        src = (perch_dir / name).with_suffix(".npz")
        dst = (dst_dir / name).with_suffix(".npy")
        if not src.exists() or dst.exists():
            skipped += 1
            continue
        curve = per_second(normalize(np.load(src)["logits"].astype(np.float32), mean, top))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part.npy")  # renamed only once fully written
        np.save(tmp, curve.astype(np.float16))
        os.replace(tmp, dst)
        done += 1
    print(f"{done} curves written, {skipped} skipped (already written or not cached)", flush=True)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perch", default=C.PERCH_DIR)
    ap.add_argument("--dst", default=C.CURVES_DIR)
    ap.add_argument("--raw", default=C.RAW_DIR)
    args = ap.parse_args()
    write_curves(args.perch, args.dst, args.raw)
