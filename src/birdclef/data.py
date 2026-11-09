"""Training items, one per (file, label), and the held-out data.

Training clips: a clip with a primary and 2 secondary labels gives 3 items. The crop is centred on a second chosen by
`selection`: "random" (uniform over the clip) or "perch" (proportional to the label's per-second normalized Perch
curve, negative values counted as 0, for trusted labels; other labels fall back to "random").
Labelled soundscapes: one item per (soundscape, label); the crop is one of the 5 s segments where the label is
annotated, chosen uniformly.
The target is 1 for the item's label; the file's other labels in the crop are masked out (mask 0), every other class
is a negative (0). With `pseudo`, every other class that Perch scores with a clip AUC above config.TRUST_AUC
(trusted.csv) gets clip(mean normalized curve over the crop, 0, 1) ** config.PSEUDO_POWER instead of 0; untrusted
classes stay 0. Targets are not normalized here; the loss normalizes them (after mixup, which takes their max).
With `embedding`, an item also holds Perch's embedding of the 1 s-stride window that starts at the crop's start
second (`has_embedding` 0 when the file has no cached Perch output), for distillation.
A dataset built with `holdout` (held-out soundscapes trained on their unscored labels) masks the `scored` classes in
every item, gets no Perch embedding target, and flags its items (`holdout` 1) so that mixes containing them get none
either.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from . import config as C

N = C.WINDOW * C.SR


def load_classes(raw_dir=C.RAW_DIR):
    return pd.read_csv(Path(raw_dir) / "taxonomy.csv").primary_label.astype(str).tolist()


def clip_items(classes, train_csv=C.RAW_DIR / "train.csv"):
    train = pd.read_csv(train_csv).astype({"primary_label": str})
    rows = []
    for f, p, s in zip(train.filename, train.primary_label, train.secondary_labels):
        labels = [c for c in dict.fromkeys([p, *ast.literal_eval(s)]) if c in classes]
        rows += [(f, c, [o for o in labels if o != c]) for c in labels]
    return pd.DataFrame(rows, columns=["filename", "label", "others"])


def soundscape_segments(labels_csv=C.RAW_DIR / "train_soundscapes_labels.csv"):
    """One row per labelled 5 s segment (the csv lists some segments more than once); start in seconds."""
    d = pd.read_csv(labels_csv)
    d["start"] = pd.to_timedelta(d.start).dt.total_seconds().astype(int)
    d["labels"] = d.primary_label.astype(str).str.split(";")
    return d.groupby(["filename", "start"]).labels.agg(lambda x: sorted(set().union(*x))).reset_index()


def soundscape_items(segments):
    rows = segments.explode("labels").groupby(["filename", "labels"]).start.agg(list).reset_index()
    return rows.rename(columns={"labels": "label", "start": "starts"})


def split(clips, segments, classes, valid=0.2, seed=0, train_csv=C.RAW_DIR / "train.csv"):
    """Random split by file. Clips whose primary label has at most config.RARE clips stay in training; those labels
    are left out of evaluation (returned boolean mask over classes)."""
    counts = pd.read_csv(train_csv).primary_label.astype(str).value_counts()
    evaluated = np.array([counts.get(c, 0) > C.RARE for c in classes])
    rng = np.random.default_rng(seed)
    primary = clips.drop_duplicates("filename").set_index("filename").label  # first label of a clip is its primary
    candidates = primary[primary.map(dict(zip(classes, evaluated)))].index.to_numpy()
    valid_clips = set(rng.choice(candidates, min(int(valid * len(primary)), len(candidates)), replace=False))
    files = segments.filename.unique()
    valid_scapes = set(rng.choice(files, max(1, int(valid * len(files))), replace=False))
    return valid_clips, valid_scapes, evaluated


class Crops(Dataset):
    def __init__(
        self,
        classes,
        subset,
        pseudo=False,
        embedding=False,
        wav_dir=C.WAV_DIR,
        curves_dir=C.CURVES_DIR,
        perch_dir=C.PERCH_DIR,
        holdout=False,
        scored=None,
    ):
        self.subset, self.pseudo, self.embedding, self.holdout = subset, pseudo, embedding, holdout
        self.scored = torch.from_numpy(np.asarray(scored, bool)) if holdout else None
        self.wav_dir, self.curves_dir, self.perch_dir = Path(wav_dir), Path(curves_dir), Path(perch_dir)
        self.index = {c: i for i, c in enumerate(classes)}
        trust = pd.read_csv(self.curves_dir / "trusted.csv").astype({"primary_label": str})
        trusted = trust[trust.trusted & trust.primary_label.isin(self.index)]
        self.curve_column = dict(zip(trusted.primary_label, trusted.index))  # column in the curve files
        self.trusted_classes = np.array([self.index[c] for c in trusted.primary_label], int)

    def path(self, filename, root, suffix):
        return (root / self.subset / filename).with_suffix(suffix)

    def curve(self, filename):
        return np.load(self.path(filename, self.curves_dir, ".npy"), mmap_mode="r")

    def crop(self, filename, start, label, others):
        path = self.path(filename, self.wav_dir, ".wav")
        wave = sf.read(path, start=start, stop=start + N, dtype="float32")[0]
        target, mask = torch.zeros(len(self.index)), torch.ones(len(self.index))
        target[self.index[label]] = 1
        mask[[self.index[o] for o in others if o in self.index]] = 0
        if self.holdout:
            mask[self.scored] = 0  # the labels this audio is evaluated on are never trained on
        if self.pseudo and len(self.trusted_classes):
            s = start // C.SR
            v = np.asarray(self.curve(filename)[s : s + C.WINDOW, list(self.curve_column.values())], np.float32)
            soft = torch.from_numpy(v.mean(0).clip(0, 1) ** C.PSEUDO_POWER)
            k = torch.from_numpy(self.trusted_classes)
            free = (mask[k] == 1) & (target[k] == 0)  # not the item's label, not masked
            target[k[free]] = soft[free]
        item = {"wave": torch.from_numpy(np.pad(wave, (0, N - len(wave)))), "target": target, "mask": mask}
        item["holdout"] = torch.tensor(float(self.holdout))
        if self.embedding:
            path = self.path(filename, self.perch_dir, ".npz")
            if path.exists() and not self.holdout:
                e = np.load(path)["embedding"]
                item["embedding"] = torch.from_numpy(e[min(round(start / C.SR), len(e) - 1)].astype(np.float32))
                item["has_embedding"] = torch.tensor(1.0)
            else:
                item["embedding"], item["has_embedding"] = torch.zeros(1536), torch.tensor(0.0)
        return item


class ClipDataset(Crops):
    def __init__(self, items, classes, selection="random", **kw):
        super().__init__(classes, "train_audio", **kw)
        assert selection in ("random", "perch"), selection
        self.items, self.selection = items.reset_index(drop=True), selection

    def __len__(self):
        return len(self.items)

    def weights(self, filename, label, seconds):
        """Sampling weight of each second of the clip as a crop centre."""
        if self.selection == "perch" and label in self.curve_column:
            w = np.asarray(self.curve(filename)[:seconds, self.curve_column[label]], np.float64).clip(0, None)
            if w.sum() > 0:
                return w / w.sum()
        return np.full(seconds, 1 / seconds)

    def __getitem__(self, i):
        r = self.items.iloc[i]
        frames = sf.info(self.path(r.filename, self.wav_dir, ".wav")).frames
        w = torch.from_numpy(self.weights(r.filename, r.label, max(1, int(np.ceil(frames / C.SR)))))
        centre = int((torch.multinomial(w, 1).item() + torch.rand(1).item()) * C.SR)  # anywhere in the chosen second
        start = min(max(centre - N // 2, 0), max(frames - N, 0))
        return {**self.crop(r.filename, start, r.label, r.others), "start": start}


class SoundscapeDataset(Crops):
    def __init__(self, items, segments, classes, **kw):
        super().__init__(classes, "train_soundscapes", **kw)
        self.items = items.reset_index(drop=True)
        self.labels = {(f, s): set(v) for f, s, v in zip(segments.filename, segments.start, segments.labels)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items.iloc[i]
        s = r.starts[torch.randint(len(r.starts), (1,)).item()]
        others = self.labels[r.filename, s] - {r.label}
        return {**self.crop(r.filename, s * C.SR, r.label, others), "start": s * C.SR}


class ClipWindows(Dataset):
    """Held-out clip as consecutive 5 s windows (the last one zero-padded), at most max_windows."""

    def __init__(self, filenames, max_windows=24, wav_dir=C.WAV_DIR):
        self.filenames, self.max_windows, self.wav_dir = list(filenames), max_windows, Path(wav_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        path = (self.wav_dir / "train_audio" / self.filenames[i]).with_suffix(".wav")
        x = sf.read(path, stop=self.max_windows * N, dtype="float32")[0]
        x = np.pad(x, (0, -len(x) % N or (N if len(x) == 0 else 0)))
        return torch.from_numpy(x.reshape(-1, N))


class SoundscapeSegments(Dataset):
    def __init__(self, segments, wav_dir=C.WAV_DIR):
        self.segments, self.wav_dir = segments.reset_index(drop=True), Path(wav_dir)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, i):
        r = self.segments.iloc[i]
        path = (self.wav_dir / "train_soundscapes" / r.filename).with_suffix(".wav")
        x = sf.read(path, start=r.start * C.SR, stop=r.start * C.SR + N, dtype="float32")[0]
        return torch.from_numpy(np.pad(x, (0, N - len(x))))


def mixup(wave, target, mask, p, generator=None):
    """Mix a fraction p of the batch with a shuffled copy: each crop is abs-max normalized, then
    lambda * x_i + (1 - lambda) * x_j with lambda uniform in config.MIXUP_LAMBDA. The target is the max of the two
    targets; a class stays masked only if it is masked in one crop and not a target in the other. Unmixed samples are
    returned unchanged. Also returns the boolean mask of mixed samples and each sample's partner (perm)."""
    n, dev = len(wave), wave.device
    order = torch.randperm(n, device=dev, generator=generator)
    perm = torch.empty_like(order)
    perm[order] = order.roll(1)  # a single cycle: every sample gets a partner other than itself
    mixed = (torch.rand(n, device=dev, generator=generator) < p) & (perm != torch.arange(n, device=dev))
    lo, hi = C.MIXUP_LAMBDA
    lam = lo + (hi - lo) * torch.rand(n, 1, device=dev, generator=generator)
    x = wave / wave.abs().amax(1, keepdim=True).clamp_min(1e-6)
    t = torch.maximum(target, target[perm])
    m = torch.maximum(torch.minimum(mask, mask[perm]), (t > 0).to(mask.dtype))
    sel = mixed[:, None]
    return (
        torch.where(sel, lam * x + (1 - lam) * x[perm], wave),
        torch.where(sel, t, target),
        torch.where(sel, m, mask),
        mixed,
        perm,
    )


class PseudoSoundscapes(Dataset):
    """Random whole-second 5 s crops of the soundscapes pseudo-labelled by pseudo.py (pseudo_dir), with target
    max over the crop's 5 seconds of softmax(frame logits), to the power `power`."""

    def __init__(self, pseudo_dir, power=1.0, wav_dir=C.WAV_DIR):
        self.files = sorted((Path(pseudo_dir) / "train_soundscapes").glob("*.npy"))
        self.power, self.wav_dir = power, Path(wav_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        logits = np.load(self.files[i], mmap_mode="r")
        s = torch.randint(max(1, len(logits) - C.WINDOW + 1), (1,)).item()
        p = torch.from_numpy(np.asarray(logits[s : s + C.WINDOW], np.float32)).softmax(1).amax(0) ** self.power
        path = self.wav_dir / "train_soundscapes" / f"{self.files[i].stem}.wav"
        wave = sf.read(path, start=s * C.SR, stop=s * C.SR + N, dtype="float32")[0]
        return {"wave": torch.from_numpy(np.pad(wave, (0, N - len(wave)))), "target": p}


class WithPseudo(Dataset):
    """Each item of `data` with a random item of `pseudo` attached (keys prefixed with pseudo_)."""

    def __init__(self, data, pseudo):
        self.data, self.pseudo = data, pseudo

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        j = torch.randint(len(self.pseudo), (1,)).item()
        return {**self.data[i], **{f"pseudo_{k}": v for k, v in self.pseudo[j].items()}}


def mix_pseudo(wave, target, mask, pseudo_wave, pseudo_target, generator=None):
    """Mix every crop with its pseudo-labelled soundscape crop: both abs-max normalized, lambda * crop +
    (1 - lambda) * soundscape with lambda uniform in config.MIXUP_LAMBDA; the target is the max of the two targets on
    the classes the crop does not mask (its masked classes stay masked)."""
    lo, hi = C.MIXUP_LAMBDA
    lam = lo + (hi - lo) * torch.rand(len(wave), 1, device=wave.device, generator=generator)
    norm = lambda x: x / x.abs().amax(1, keepdim=True).clamp_min(1e-6)  # noqa: E731
    return lam * norm(wave) + (1 - lam) * norm(pseudo_wave), torch.maximum(target, pseudo_target) * mask
