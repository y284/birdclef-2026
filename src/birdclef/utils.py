import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def macro_auc(y, scores):
    """Mean ROC-AUC over the classes with at least one positive and one negative; also returns the per-class AUC
    (NaN for the classes skipped)."""
    y = np.asarray(y)
    per_class = np.full(y.shape[1], np.nan)
    for c in np.flatnonzero((y.sum(0) > 0) & (y.sum(0) < len(y))):
        per_class[c] = roc_auc_score(y[:, c], scores[:, c])
    return (np.nanmean(per_class) if np.isfinite(per_class).any() else np.nan), per_class


def devices():
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]


def amp_dtype(device):
    """bfloat16 where the GPU supports it natively (compute capability >= 8), float16 otherwise (e.g. T4)."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def autocast(device):
    return torch.autocast(device.type, dtype=amp_dtype(device), enabled=device.type == "cuda")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_on_devices(jobs, fn):
    """Returns [fn(device, job) for job in jobs]. With 2+ GPUs, each GPU takes every n-th job in its own thread."""
    devs = devices()
    if len(devs) == 1:
        return [fn(devs[0], job) for job in jobs]
    out = [None] * len(jobs)

    def work(i):
        for k in range(i, len(jobs), len(devs)):
            out[k] = fn(devs[i], jobs[k])

    with ThreadPoolExecutor(len(devs)) as pool:
        list(pool.map(work, range(len(devs))))
    return out
