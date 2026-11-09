import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as C
from .data import ClipWindows, SoundscapeSegments
from .utils import autocast, macro_auc


def smooth(frames):
    """[..., steps, classes] -> step i + (step i-1 + step i+1) / 2, along the steps (seconds for pseudo-labels, 5 s
    windows for a submission); the first and last steps count themselves as their missing neighbour."""
    prev = torch.cat([frames[..., :1, :], frames[..., :-1, :]], -2)
    nxt = torch.cat([frames[..., 1:, :], frames[..., -1:, :]], -2)
    return frames + (prev + nxt) / 2


def to_windows(frames, seconds=C.WINDOW):
    """[..., seconds, classes] per-second frame logits -> [..., windows, classes], max over each 5 s window."""
    n = frames.shape[-2] // seconds * seconds
    return frames[..., :n, :].unflatten(-2, (-1, seconds)).amax(-2)


def submission_scores(frames):
    """[..., seconds, classes] frame logits of a whole soundscape -> [..., windows, classes] scores of its submission
    rows: max over each 5 s window, then smoothed with the neighbouring windows."""
    return smooth(to_windows(frames))


@torch.no_grad()
def predict(model, waves, device, batch=128):
    """[n, samples] -> [n, classes] clip logits."""
    out = []
    for i in range(0, len(waves), batch):
        with autocast(device):
            out.append(model(waves[i : i + batch].to(device, non_blocking=True))[0].float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def clip_scores(model, filenames, device, max_windows=24, batch=128, workers=2, wav_dir=None):
    """Per held-out clip: the max logit of each class over its 5 s windows."""
    ds = ClipWindows(filenames, max_windows, **({"wav_dir": wav_dir} if wav_dir else {}))
    scores, pending, counts = [], [], []

    def flush():
        logits = predict(model, torch.cat(pending), device, batch)
        for part in np.split(logits, np.cumsum(counts)[:-1]):
            scores.append(part.max(0))
        pending.clear()
        counts.clear()

    for windows in DataLoader(ds, batch_size=None, num_workers=workers):
        pending.append(windows)
        counts.append(len(windows))
        if sum(counts) >= batch:
            flush()
    if pending:
        flush()
    return np.stack(scores)


@torch.no_grad()
def soundscape_scores(model, segments, device, batch=128, workers=2, wav_dir=None):
    ds = SoundscapeSegments(segments, **({"wav_dir": wav_dir} if wav_dir else {}))
    out = [predict(model, x, device, batch) for x in DataLoader(ds, batch, num_workers=workers)]
    return np.concatenate(out)


def evaluate(model, valid, device, args):
    """Macro ROC-AUC on the held-out clips (clip level) and soundscape segments, over the evaluated classes."""
    model.eval()
    kw = {"batch": args.batch_size * 2, "workers": args.workers, "wav_dir": args.wav}
    clip = clip_scores(model, valid["clips"], device, args.max_windows, **kw)
    scape = soundscape_scores(model, valid["segments"], device, **kw)
    model.train()
    e = valid["evaluated"]
    res = {
        "clip_auc": macro_auc(valid["clip_y"][:, e], clip[:, e])[0],
        "soundscape_auc": macro_auc(valid["segment_y"][:, e], scape[:, e])[0],
    }
    res["score"] = float(np.nanmean(list(res.values())))
    return res
