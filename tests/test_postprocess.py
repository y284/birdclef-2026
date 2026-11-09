import numpy as np

from birdclef.postprocess import normalize, per_second, trusted_labels, write_curves


def test_per_second_weights_by_position():
    x = np.zeros((3, 1))
    x[1] = 1  # window 1 covers seconds 1..5
    y = per_second(x).ravel()
    assert len(y) == 3 + 4
    # second 1: window 0 at position 1 (0.3), window 1 at position 0 (0.1) -> 0.1 / 0.4
    # second 3: windows 0, 1, 2 at positions 3, 2, 1 (0.3, 0.6, 0.3) -> 0.6 / 1.2
    assert np.allclose(y, [0, 0.25, 0.3, 0.5, 0.3, 0.25, 0])
    assert np.allclose(per_second(np.full((56, 2), 3.0)), 3.0)  # a constant stays constant, edges included


def test_normalize_maps_mean_to_0_and_top_to_1():
    mean, top = np.array([1.0, -2.0]), np.array([3.0, 2.0])
    assert np.allclose(normalize(np.array([[1.0, -2.0], [3.0, 2.0]]), mean, top), [[0, 0], [1, 1]])


def test_write_curves_only_for_labelled_files(tmp_path):
    perch, raw, dst = tmp_path / "perch", tmp_path / "raw", tmp_path / "curves"
    rng = np.random.default_rng(0)
    for name, n in [("train_audio/a/1", 4), ("train_soundscapes/s1", 56), ("train_soundscapes/s2", 56)]:
        (perch / name).parent.mkdir(parents=True, exist_ok=True)
        np.savez((perch / name).with_suffix(".npz"), logits=rng.normal(size=(n, 2)).astype(np.float16))
    (perch / "perch_map.csv").write_text("primary_label,perch_index\na,0\nb,1\n")
    raw.mkdir()
    (raw / "train.csv").write_text("primary_label,secondary_labels,filename\na,['b'],a/1.ogg\n")
    (raw / "train_soundscapes_labels.csv").write_text("filename,start,end,primary_label\ns1.ogg,00:00:00,00:00:05,a\n")

    assert write_curves(perch, dst, raw) == 2
    assert np.load(dst / "train_audio/a/1.npy").shape == (4 + 4, 2)
    assert np.load(dst / "train_soundscapes/s1.npy").shape == (60, 2)
    assert not (dst / "train_soundscapes/s2.npy").exists()  # unlabelled soundscape
    assert (dst / "norm_stats.npz").exists() and (dst / "trusted.csv").exists()
    assert write_curves(perch, dst, raw) == 0


def test_trusted_labels_from_clip_auc(tmp_path):
    rng = np.random.default_rng(0)
    rows = ["primary_label,secondary_labels,filename"]
    for i in range(40):
        primary = "a" if i % 2 else "b"
        secondary = "['a']" if i % 4 == 0 else "[]"  # a is positive in odd clips and as a secondary label
        logits = rng.normal(size=(5, 3))
        logits[2, 0] += 10 * (primary == "a" or secondary != "[]")  # Perch finds a in one window of its clips
        path = tmp_path / "train_audio" / primary / f"{i}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, logits=logits.astype(np.float16))
        rows.append(f'{primary},"{secondary}",{primary}/{i}.ogg')
    (tmp_path / "train.csv").write_text("\n".join(rows) + "\n")

    t = trusted_labels(["a", "b", "c"], tmp_path, tmp_path / "train.csv")
    assert t.auc[0] == 1.0 and t.trusted[0]  # a: always found
    assert 0.2 < t.auc[1] < 0.8 and not t.trusted[1]  # b: random scores
    assert np.isnan(t.auc[2]) and not t.trusted[2]  # c: no positive clip
