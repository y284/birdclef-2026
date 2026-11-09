"""Pseudo-labels for the unlabelled soundscapes (not in train_soundscapes_labels.csv). Every model runs on the whole
recording at once, giving one frame per second; frame logits are averaged over the models and smoothed with the
neighbouring seconds (infer.smooth). One float16 .npy [seconds, classes] per soundscape in --out, columns in
taxonomy.csv order; models.json lists the checkpoints. Files already written are skipped.

    python -m birdclef.pseudo --models outputs/v2/model_A_seed42_best.pt outputs/v2/model_A_seed43_best.pt \
        --out data/pseudo/iter1
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .data import load_classes
from .infer import smooth
from .model import Net
from .utils import autocast, devices


def unlabelled_soundscapes(raw_dir=C.RAW_DIR, wav_dir=C.WAV_DIR):
    labelled = {Path(f).stem for f in pd.read_csv(Path(raw_dir) / "train_soundscapes_labels.csv").filename}
    return sorted(p.stem for p in (Path(wav_dir) / "train_soundscapes").glob("*.wav") if p.stem not in labelled)


class Recordings(Dataset):
    def __init__(self, names, wav_dir):
        self.names, self.wav_dir = names, Path(wav_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        x = sf.read(self.wav_dir / "train_soundscapes" / f"{self.names[i]}.wav", dtype="float32")[0]
        x = x[: len(x) // C.SR * C.SR]  # whole seconds
        return self.names[i], torch.from_numpy(x)


def load_models(paths, n_classes, backbone, device):
    models = []
    for path in paths:
        model = Net(n_classes, backbone, pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models.append(model.to(device).eval())
    return models


@torch.no_grad()
def frame_logits(models, waves, device):
    """[B, samples] -> [B, seconds, classes] frame logits, averaged over the models."""
    waves = waves.to(device, non_blocking=True)
    total = 0
    for model in models:
        with autocast(device):
            total = total + model(waves)[1].float()
    return total / len(models)


def save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.npy")  # renamed only once fully written
    np.save(tmp, array)
    os.replace(tmp, path)


def recording_logits(models, names, device, wav_dir=C.WAV_DIR, batch=16, workers=4):
    """Yields (name, [seconds, classes] frame logits averaged over the models) for every recording."""
    pending = []

    def flush():
        logits = frame_logits(models, torch.stack([x for _, x in pending]), device).cpu()
        yield from zip([n for n, _ in pending], logits)
        pending.clear()

    for i, (name, x) in enumerate(DataLoader(Recordings(names, wav_dir), batch_size=None, num_workers=workers), 1):
        if pending and len(x) != len(pending[0][1]):  # a batch holds recordings of one length
            yield from flush()
        pending.append((name, x))
        if len(pending) == batch:
            yield from flush()
        if i % 500 == 0:
            print(f"{i}/{len(names)} soundscapes", flush=True)
    if pending:
        yield from flush()


def pseudo_label(models, names, out, device, wav_dir=C.WAV_DIR, batch=16, workers=4):
    out = Path(out) / "train_soundscapes"
    todo = [n for n in names if not (out / f"{n}.npy").exists()]
    for name, logits in recording_logits(models, todo, device, wav_dir, batch, workers):
        save(out / f"{name}.npy", smooth(logits).half().numpy())
    print(f"{len(todo)} written, {len(names) - len(todo)} already done", flush=True)
    return len(todo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", type=Path, required=True, help="checkpoints, averaged")
    p.add_argument("--out", type=Path, required=True, help="e.g. data/pseudo/iter1")
    p.add_argument("--backbone", default="tf_efficientnet_b0.ns_jft_in1k")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--raw", type=Path, default=C.RAW_DIR)
    p.add_argument("--wav", type=Path, default=C.WAV_DIR)
    args = p.parse_args()
    device = devices()[0]
    classes = load_classes(args.raw)
    names = unlabelled_soundscapes(args.raw, args.wav)
    print(f"{len(names)} unlabelled soundscapes, {len(args.models)} models", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "models.json").write_text(
        json.dumps({"models": [str(m) for m in args.models], "backbone": args.backbone, "classes": classes}, indent=2)
    )
    pseudo_label(
        load_models(args.models, len(classes), args.backbone, device),
        names,
        args.out,
        device,
        args.wav,
        args.batch,
        args.workers,
    )


if __name__ == "__main__":
    main()
