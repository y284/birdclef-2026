import numpy as np
import pandas as pd
import pytest
import torch

from birdclef import config as C
from birdclef.export import export_onnx, onnxruntime_runner, openvino_runner, segment_scores, torch_runner
from birdclef.model import LogMel, Net


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = Net(7, "tf_efficientnet_b0", pretrained=False).eval()
    path = export_onnx(model, tmp_path_factory.mktemp("onnx") / "model.onnx", seconds=10)
    wave = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 10 * C.SR)).astype(np.float32))
    with torch.no_grad():
        mel = LogMel()(wave).numpy()
    return model, path, mel


def test_onnxruntime_matches_pytorch(exported):
    model, path, mel = exported
    ref = torch_runner([model], 2)(mel)
    assert ref.shape == (1, 10, 7)
    assert np.abs(onnxruntime_runner([path], 2)(mel) - ref).max() < 1e-4
    assert np.abs(onnxruntime_runner([path], 2, fp16=True)(mel) - ref).max() < 1e-2


def test_openvino_matches_pytorch(exported):
    pytest.importorskip("openvino")
    model, path, mel = exported
    ref = torch_runner([model], 2)(mel)
    assert np.abs(openvino_runner([path], 2)(mel) - ref).max() < 1e-4
    assert np.abs(openvino_runner([path], 2, "f16")(mel) - ref).max() < 1e-2


def test_segment_scores_pick_the_window_of_each_segment():
    frames = {"a.ogg": np.arange(10 * 2, dtype=np.float32).reshape(10, 2)}  # 2 windows of 5 s
    held = pd.DataFrame({"filename": ["a.ogg", "a.ogg"], "start": [5, 0]})
    windows = segment_scores(frames, held, smoothed=False)
    assert (windows == [[18, 19], [8, 9]]).all()  # max over seconds 5-9, then 0-4
    smoothed = segment_scores(frames, held, smoothed=True)
    assert (smoothed == [[18 + (8 + 18) / 2, 19 + (9 + 19) / 2], [8 + (8 + 18) / 2, 9 + (9 + 19) / 2]]).all()
