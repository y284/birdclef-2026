"""Site and hour priors. Every train soundscape (labelled or not) is scored by the ensemble as a submission is
(infer.submission_scores), and the scores are turned into probabilities (sigmoid, and softmax over the classes). Each
species' mean probability over the recordings of a site, and over the recordings of an hour of the day, multiplies
its probabilities in every soundscape of that site and hour (the mean over all recordings stands in for a site or hour
never seen). No labels are used, so the same runs on the test soundscapes of a submission. Reports the held-out
soundscape macro ROC-AUC before and after.

    python -m birdclef.sitehour --models outputs/iter1/model_A_seed42.pt outputs/iter1/model_A_seed43.pt
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .export import held_out
from .infer import submission_scores
from .pseudo import load_models, recording_logits
from .utils import devices, macro_auc

NAME = re.compile(r"(?P<site>[A-Z]+\d+)_(?P<date>\d{8})_(?P<hour>\d{2})\d{4}")


def site_hour(name):
    """'..._S08_20250606_030007' -> ('S08', 3)."""
    m = NAME.search(Path(name).stem)
    return m["site"], int(m["hour"])


def probabilities(scores, kind="sigmoid"):
    return torch.sigmoid(scores) if kind == "sigmoid" else scores.softmax(-1)


def priors(probs, keys):
    """probs [recordings, windows, classes], keys [recordings] -> {key: mean probability per class}, plus the mean
    over all recordings under None."""
    means = probs.mean(1)
    out = {k: means[keys == k].mean(0) for k in np.unique(keys)}
    out[None] = means.mean(0)
    return out


def apply(probs, names, site_prior=None, hour_prior=None):
    """Multiplies each recording's probabilities by the priors of its site and hour."""
    out = probs.copy()
    for i, (site, hour) in enumerate(map(site_hour, names)):
        if site_prior is not None:
            out[i] *= site_prior.get(site, site_prior[None])
        if hour_prior is not None:
            out[i] *= hour_prior.get(hour, hour_prior[None])
    return out


def soundscape_scores(models, names, device, wav_dir, batch, workers):
    """[recordings, windows, classes] submission scores (logits) of every recording, in the order of names."""
    scores = {
        n: submission_scores(x).numpy() for n, x in recording_logits(models, names, device, wav_dir, batch, workers)
    }
    return np.stack([scores[n] for n in names])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", type=Path, required=True, help="checkpoints, frame logits averaged")
    p.add_argument("--backbone", default="tf_efficientnet_b0.ns_jft_in1k")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--valid", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--raw", type=Path, default=C.RAW_DIR)
    p.add_argument("--wav", type=Path, default=C.WAV_DIR)
    p.add_argument(
        "--cache", type=Path, default=Path("outputs/soundscape_scores.npz"), help="scores, reused for the same models"
    )
    p.add_argument("--out", type=Path, default=Path("results/site_hour.json"))
    args = p.parse_args()

    held, y, evaluated, classes = held_out(args.raw, args.valid, args.split_seed)
    names = sorted(f.stem for f in (args.wav / "train_soundscapes").glob("*.wav"))
    models = [str(m) for m in args.models]
    cached = np.load(args.cache) if args.cache.exists() else None
    if cached is not None and "models" in cached.files and cached["models"].tolist() == models:
        names, scores = cached["names"].tolist(), cached["scores"].astype(np.float32)
    else:
        device = devices()[0]
        nets = load_models(args.models, len(classes), args.backbone, device)
        scores = soundscape_scores(nets, names, device, args.wav, args.batch, args.workers)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, models=np.array(models), names=np.array(names), scores=scores.astype(np.float16))

    sites, hours = map(np.array, zip(*map(site_hour, names)))
    index = {n: i for i, n in enumerate(names)}
    rows = [index[Path(f).stem] for f in held.filename]
    windows = (held.start // C.WINDOW).to_numpy()

    def auc(p):
        return float(macro_auc(y[:, evaluated], p[rows, windows][:, evaluated])[0])

    report = {
        "models": models,
        "soundscapes": len(names),
        "sites": len(np.unique(sites)),
        "hours": len(np.unique(hours)),
        "auc": {},
    }
    for kind in ("sigmoid", "softmax"):
        probs = probabilities(torch.from_numpy(scores), kind).numpy()
        site_prior, hour_prior = priors(probs, sites), priors(probs, hours)
        report["auc"][kind] = {
            "none": auc(probs),
            "site": auc(apply(probs, names, site_prior)),
            "hour": auc(apply(probs, names, hour_prior=hour_prior)),
            "site_and_hour": auc(apply(probs, names, site_prior, hour_prior)),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()