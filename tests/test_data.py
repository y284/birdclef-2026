import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from birdclef import config as C
from birdclef.data import (
    ClipDataset,
    SoundscapeDataset,
    clip_items,
    mixup,
    soundscape_items,
    soundscape_segments,
    split,
)

CLASSES = ["a", "b", "c", "d"]


@pytest.fixture
def data(tmp_path):
    wav, curves = tmp_path / "wav", tmp_path / "curves"
    for name, seconds in [
        ("train_audio/a/long.wav", 30),
        ("train_audio/a/short.wav", 2),
        ("train_soundscapes/s.wav", 60),
    ]:
        (wav / name).parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav / name, np.full(seconds * C.SR, 0.5), C.SR, subtype="PCM_16")
    curve = np.full((34, 3), -0.5, np.float16)  # 30 windows + 4 seconds; columns a, b, c (d not in Perch)
    curve[20, 0] = 2.0  # a calls at second 20 only
    for name, c in [("train_audio/a/long", curve), ("train_audio/a/short", curve[:6]), ("train_soundscapes/s", curve)]:
        (curves / name).parent.mkdir(parents=True, exist_ok=True)
        np.save((curves / name).with_suffix(".npy"), c)
    trust = pd.DataFrame({"primary_label": ["a", "b", "c"], "auc": [0.99, 0.95, 0.5], "trusted": [True, True, False]})
    trust.to_csv(curves / "trusted.csv", index=False)
    csv = tmp_path / "train.csv"
    csv.write_text("primary_label,secondary_labels,filename\na,\"['b', 'd']\",a/long.ogg\na,[],a/short.ogg\n")
    (tmp_path / "labels.csv").write_text(
        "filename,start,end,primary_label\n"
        "s.ogg,00:00:00,00:00:05,a;b\ns.ogg,00:00:00,00:00:05,a;b\n"  # the competition file repeats segments
        "s.ogg,00:00:15,00:00:20,b;c\n"
    )
    return clip_items(CLASSES, csv), soundscape_segments(tmp_path / "labels.csv"), wav, curves


def test_one_item_per_label_with_the_others_masked(data):
    items, _, wav, curves = data
    assert items.label.tolist() == ["a", "b", "d", "a"]
    x = ClipDataset(items, CLASSES, "random", wav_dir=wav, curves_dir=curves)[1]  # item for secondary label b
    assert x["target"].tolist() == [0, 1, 0, 0]
    assert x["mask"].tolist() == [0, 1, 1, 0]  # a and d are labelled in the clip but maybe not in this crop
    assert x["wave"].shape == (C.WINDOW * C.SR,)


def test_perch_selection_centres_on_the_label_curve(data):
    items, _, wav, curves = data
    ds = ClipDataset(items, CLASSES, "perch", wav_dir=wav, curves_dir=curves)
    torch.manual_seed(0)
    starts = [ds[0]["start"] for _ in range(20)]  # a: all weight on second 20 -> centre in [20, 21) s
    assert all(17.5 * C.SR <= s <= 18.5 * C.SR for s in starts)
    random = ClipDataset(items, CLASSES, "random", wav_dir=wav, curves_dir=curves)
    assert np.std([random[0]["start"] for _ in range(50)]) > 3 * C.SR
    assert np.allclose(ds.weights("a/long", "b", 30), 1 / 30)  # trusted, but no positive value: random
    assert np.allclose(ds.weights("a/long", "c", 30), 1 / 30)  # in Perch, not trusted: random
    assert np.allclose(ds.weights("a/long", "d", 30), 1 / 30)  # not in Perch: random


def test_short_clip_is_zero_padded(data):
    items, _, wav, curves = data
    x = ClipDataset(items, CLASSES, "perch", wav_dir=wav, curves_dir=curves)[3]
    assert x["start"] == 0
    assert torch.allclose(x["wave"][: 2 * C.SR], torch.tensor(0.5), atol=1e-3)
    assert (x["wave"][2 * C.SR :] == 0).all()


def test_soundscape_items_crop_a_segment_of_their_label(data):
    _, segments, wav, curves = data
    assert segments.start.tolist() == [0, 15] and segments.labels.tolist() == [["a", "b"], ["b", "c"]]
    items = soundscape_items(segments)
    assert sorted(zip(items.label, items.starts.map(tuple))) == [("a", (0,)), ("b", (0, 15)), ("c", (15,))]
    ds = SoundscapeDataset(items, segments, CLASSES, wav_dir=wav, curves_dir=curves)
    c = ds[items.label.tolist().index("c")]
    assert c["start"] == 15 * C.SR
    assert c["target"].tolist() == [0, 0, 1, 0] and c["mask"].tolist() == [1, 0, 1, 1]  # b shares the segment


def test_pseudo_targets_are_squared_curves_of_trusted_species(data):
    _, segments, wav, curves = data
    curve = np.full((60, 3), -0.5, np.float16)  # columns a, b (trusted), c (untrusted)
    curve[15:20, 0] = 0.8  # a in the segment at 15 s
    curve[0:5, 2] = 0.9  # c in the segment at 0 s
    np.save(curves / "train_soundscapes" / "s.npy", curve)
    items = soundscape_items(segments)
    ds = SoundscapeDataset(items, segments, CLASSES, pseudo=True, wav_dir=wav, curves_dir=curves)
    c = ds[items.label.tolist().index("c")]  # segment 15 s: labels b, c
    assert torch.allclose(c["target"], torch.tensor([0.64, 0, 1, 0]), atol=1e-3)  # b is masked
    a = ds[items.label.tolist().index("a")]  # segment 0 s: labels a, b
    assert a["target"].tolist() == [1, 0, 0, 0]  # c is untrusted, d is not in Perch: both stay 0


def test_split_keeps_rare_labels_in_training(tmp_path):
    rows = ["primary_label,secondary_labels,filename"] + [f"a,[],a/{i}.ogg" for i in range(30)] + ["b,[],b/0.ogg"]
    (tmp_path / "train.csv").write_text("\n".join(rows) + "\n")
    items = clip_items(["a", "b"], tmp_path / "train.csv")
    segments = pd.DataFrame({"filename": [f"s{i}.ogg" for i in range(10)], "start": 0, "labels": [["a"]] * 10})
    valid_clips, valid_scapes, evaluated = split(items, segments, ["a", "b"], 0.2, 0, tmp_path / "train.csv")
    assert evaluated.tolist() == [True, False]  # b has 1 clip (<= C.RARE)
    assert "b/0.ogg" not in valid_clips and len(valid_clips) == 6 and len(valid_scapes) == 2


def test_embedding_is_perch_window_at_crop_start(data, tmp_path):
    items, segments, wav, curves = data
    perch = tmp_path / "perch"
    for name, windows in [("train_audio/a/long", 26), ("train_soundscapes/s", 56)]:
        (perch / name).parent.mkdir(parents=True, exist_ok=True)
        e = np.repeat(np.arange(windows, dtype=np.float16)[:, None], 1536, 1)  # window k holds k
        np.savez((perch / name).with_suffix(".npz"), embedding=e)
    kw = {"embedding": True, "wav_dir": wav, "curves_dir": curves, "perch_dir": perch}
    ds = ClipDataset(items, CLASSES, "random", **kw)
    for _ in range(10):
        x = ds[0]
        assert x["has_embedding"] == 1 and x["embedding"][0] == round(x["start"] / C.SR)
    assert ds[3]["has_embedding"] == 0  # a/short has no Perch output
    scape = soundscape_items(segments)
    x = SoundscapeDataset(scape, segments, CLASSES, **kw)[scape.label.tolist().index("c")]
    assert x["embedding"][0] == 15  # segment at 15 s


def test_mixup_targets_max_and_mask_rule():
    wave = torch.stack([torch.full((8,), 0.5), torch.full((8,), -2.0)])
    target = torch.tensor([[1.0, 0, 0, 0.3], [0, 1.0, 0, 0]])
    mask = torch.tensor([[1.0, 0, 1, 1], [1.0, 1, 0, 1]])  # class 1 masked in crop 0, class 2 in crop 1
    w, t, m, mixed, _ = mixup(wave, target, mask, p=1.0, generator=torch.Generator().manual_seed(0))
    assert mixed.all()  # a batch of 2 is always swapped
    assert torch.allclose(t, torch.tensor([[1.0, 1, 0, 0.3]] * 2))
    assert m.tolist() == [[1.0, 1, 0, 1]] * 2  # class 1 is a target of crop 1; class 2 stays masked
    # after abs-max, crop 0 is +1 and crop 1 is -1: the mixes are 2 * lambda - 1 and 1 - 2 * lambda, in [-0.4, 0.4]
    assert (w.abs() <= 0.4 + 1e-6).all() and torch.allclose(w, w[:, :1].expand(2, 8))


def test_mixup_pairs_every_sample_with_another():
    g = torch.Generator().manual_seed(0)
    wave = torch.eye(16)  # sample i is a pulse at position i
    for _ in range(20):
        w, *_, mixed, _ = mixup(wave, torch.eye(16), torch.ones(16, 16), p=1.0, generator=g)
        assert mixed.all()
        assert ((w.diagonal() >= 0.3 - 1e-6) & (w.diagonal() <= 0.7 + 1e-6)).all()  # own weight lambda
        assert ((w > 0).sum(1) == 2).all()  # plus exactly one other sample


def test_mixup_with_p_0_changes_nothing():
    wave, target, mask = torch.randn(4, 8), torch.rand(4, 3), torch.ones(4, 3)
    w, t, m, mixed, _ = mixup(wave, target, mask, p=0.0)
    assert not mixed.any() and torch.equal(w, wave) and torch.equal(t, target) and torch.equal(m, mask)


def test_holdout_items_mask_the_scored_classes_and_get_no_embedding(data, tmp_path):
    _, segments, wav, curves = data
    perch = tmp_path / "perch"
    (perch / "train_soundscapes").mkdir(parents=True)
    np.savez(perch / "train_soundscapes" / "s.npz", embedding=np.ones((56, 1536), np.float16))
    items = soundscape_items(segments)
    items = items[items.label == "c"]  # c is not scored; a and b are
    ds = SoundscapeDataset(
        items,
        segments,
        CLASSES,
        embedding=True,
        wav_dir=wav,
        curves_dir=curves,
        perch_dir=perch,
        holdout=True,
        scored=[True, True, False, False],
    )
    x = ds[0]
    assert x["target"].tolist() == [0, 0, 1, 0] and x["mask"].tolist() == [0, 0, 1, 1]
    assert x["holdout"] == 1 and x["has_embedding"] == 0
