import numpy as np
import soundfile as sf
import torch

from birdclef import config as C
from birdclef.infer import smooth, submission_scores, to_windows
from birdclef.model import Net
from birdclef.pseudo import load_models, pseudo_label, unlabelled_soundscapes


def test_smooth_adds_half_of_each_neighbour():
    f = torch.tensor([[1.0], [2.0], [4.0]])
    # first frame: 1 + (1 + 2) / 2; middle: 2 + (1 + 4) / 2; last: 4 + (2 + 4) / 2
    assert smooth(f).ravel().tolist() == [2.5, 4.5, 7.0]


def test_to_windows_takes_the_max_of_each_5_seconds():
    f = torch.arange(12.0).reshape(12, 1)
    assert to_windows(f).ravel().tolist() == [4.0, 9.0]  # seconds 10-11 do not fill a window


def test_submission_smooths_the_5_second_windows_not_the_seconds():
    f = torch.zeros(15, 1)
    f[2] = 4.0  # one loud second in the first window
    # windows: max -> [4, 0, 0]; smoothed: [4 + (4 + 0) / 2, 0 + (4 + 0) / 2, 0 + (0 + 0) / 2]
    assert submission_scores(f).ravel().tolist() == [6.0, 2.0, 0.0]


def test_pseudo_labels_unlabelled_soundscapes_at_one_frame_per_second(tmp_path):
    raw, wav, out = tmp_path / "raw", tmp_path / "wav", tmp_path / "pseudo"
    raw.mkdir()
    (raw / "train_soundscapes_labels.csv").write_text(
        "filename,start,end,primary_label\nlabelled.ogg,00:00:00,00:00:05,a\n"
    )
    rng = np.random.default_rng(0)
    for name, seconds in [("labelled", 10), ("s1", 10), ("s2", 10), ("s3", 7)]:
        (wav / "train_soundscapes").mkdir(parents=True, exist_ok=True)
        sf.write(wav / "train_soundscapes" / f"{name}.wav", 0.1 * rng.standard_normal(seconds * C.SR), C.SR)
    names = unlabelled_soundscapes(raw, wav)
    assert names == ["s1", "s2", "s3"]

    torch.manual_seed(0)
    paths = []
    for k in range(2):
        path = tmp_path / f"m{k}.pt"
        torch.save(Net(4, "tf_efficientnet_b0", pretrained=False).state_dict(), path)
        paths.append(path)
    device = torch.device("cpu")
    models = load_models(paths, 4, "tf_efficientnet_b0", device)
    assert pseudo_label(models, names, out, device, wav, batch=2, workers=0) == 3
    s1 = np.load(out / "train_soundscapes" / "s1.npy")
    assert s1.shape == (10, 4) and s1.dtype == np.float16
    assert np.load(out / "train_soundscapes" / "s3.npy").shape == (7, 4)

    x = torch.from_numpy(sf.read(wav / "train_soundscapes" / "s1.wav", dtype="float32")[0])[None]
    with torch.no_grad():
        expected = smooth((models[0](x)[1] + models[1](x)[1]) / 2)[0].numpy()
    assert np.allclose(s1, expected, atol=1e-2)  # mean of the models, then smoothed
    assert pseudo_label(models, names, out, device, wav, batch=2, workers=0) == 0


def test_pseudo_crop_target_is_max_softmax_over_5_seconds(tmp_path):
    from birdclef.data import PseudoSoundscapes

    logits = np.full((6, 3), -10.0, np.float16)
    logits[:, 0] = 10  # class 0 everywhere
    logits[2, 1] = 30  # class 1 dominates second 2 only
    (tmp_path / "p" / "train_soundscapes").mkdir(parents=True)
    np.save(tmp_path / "p" / "train_soundscapes" / "s.npy", logits)
    (tmp_path / "w" / "train_soundscapes").mkdir(parents=True)
    sf.write(tmp_path / "w" / "train_soundscapes" / "s.wav", np.full(6 * C.SR, 0.5), C.SR, subtype="PCM_16")
    for power, expected in [(1.0, [1.0, 1.0, 0.0]), (2.0, [1.0, 1.0, 0.0])]:
        x = PseudoSoundscapes(tmp_path / "p", power, tmp_path / "w")[0]
        assert x["wave"].shape == (C.WINDOW * C.SR,)
        assert torch.allclose(x["target"], torch.tensor(expected), atol=1e-3)  # every 5 s crop holds second 2
    logits[2, 1] = 10.0  # class 0 and 1 tie at second 2: softmax 0.5 each
    np.save(tmp_path / "p" / "train_soundscapes" / "s.npy", logits)
    x = PseudoSoundscapes(tmp_path / "p", 2.0, tmp_path / "w")[0]
    assert torch.allclose(x["target"], torch.tensor([1.0, 0.25, 0.0]), atol=1e-3)  # 0.5 ** 2


def test_mix_pseudo_keeps_the_max_target():
    from birdclef.data import mix_pseudo

    wave, pseudo = torch.full((2, 8), 2.0), torch.full((2, 8), -0.5)
    target, pseudo_target = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]), torch.tensor([[0.2, 0.9, 0.1]] * 2)
    mask = torch.tensor([[1.0, 1, 1], [1.0, 1, 0]])  # class 2 is another label of crop 1: unknown, stays masked
    w, t = mix_pseudo(wave, target, mask, pseudo, pseudo_target)
    assert torch.allclose(t, torch.tensor([[1.0, 0.9, 0.1], [0.2, 1.0, 0.0]]))
    assert (w.abs() <= 0.4 + 1e-6).all()  # +1 and -1 after abs-max, lambda in [0.3, 0.7]
