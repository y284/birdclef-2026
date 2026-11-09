import numpy as np
import onnx
import pytest
import soundfile as sf
from onnx import TensorProto, helper

from birdclef import config as C
from birdclef.perch import (
    Perch,
    PerchEmbedding,
    build_map,
    cache_all,
    norm_stats,
    prefetch,
    providers,
    split_perch,
    windows,
)

N = C.WINDOW * C.SR


def fake_perch(path, n_labels=7):
    """Same input / output names as Perch v2; embedding and label are the first samples of each window."""
    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    e_end = helper.make_tensor("e_end", TensorProto.INT64, [1], [1536])
    l_end = helper.make_tensor("l_end", TensorProto.INT64, [1], [n_labels])
    graph = helper.make_graph(
        [
            helper.make_node("Slice", ["inputs", "starts", "e_end", "axes"], ["embedding"]),
            helper.make_node("Slice", ["inputs", "starts", "l_end", "axes"], ["label"]),
        ],
        "fake_perch",
        [helper.make_tensor_value_info("inputs", TensorProto.FLOAT, ["batch", N])],
        [
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 1536]),
            helper.make_tensor_value_info("label", TensorProto.FLOAT, ["batch", n_labels]),
        ],
        [starts, axes, e_end, l_end],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), path)
    return path


@pytest.mark.parametrize("seconds, count", [(3, 1), (5, 1), (7.5, 3), (60, 56)])
def test_window_count_and_offsets(seconds, count):
    x = np.arange(int(seconds * C.SR), dtype=np.float32)
    w = windows(x)
    assert w.shape == (count, N)
    assert np.array_equal(w[:, 0], np.arange(count) * C.STRIDE * C.SR)


def test_cache_aligns_windows_with_files(tmp_path):
    wav, out = tmp_path / "wav", tmp_path / "perch"
    lengths = {"train_soundscapes/s.wav": 12, "train_audio/sp1/a.wav": 2, "train_audio/sp1/b.wav": 6}
    rng = np.random.default_rng(0)
    signals = {}
    for name, seconds in lengths.items():
        (wav / name).parent.mkdir(parents=True, exist_ok=True)
        x = rng.uniform(-0.9, 0.9, seconds * C.SR)
        sf.write(wav / name, x, C.SR, subtype="PCM_16")
        signals[name] = sf.read(wav / name, dtype="float32")[0]  # the 16-bit values the model sees

    columns = [3, 0]  # our first kept class is Perch column 3, the second is column 0
    model = Perch(columns, fake_perch(tmp_path / "fake.onnx"))
    assert cache_all(model, wav, out, batch=3) == 3  # batches span file boundaries
    for name, x in signals.items():
        z = np.load(out / name.replace(".wav", ".npz"))
        n = max(1, len(x) // C.SR - C.WINDOW + 1)
        assert z["embedding"].shape == (n, 1536) and z["logits"].shape == (n, 2)
        assert z["logits"].dtype == np.float16
        k = np.arange(n) * C.SR
        assert np.allclose(z["logits"][:, 1], x[k], atol=1e-3)  # window k starts at second k of its own file
        assert np.allclose(z["logits"][:, 0], x[k + 3], atol=1e-3)  # columns follow the given order
    assert cache_all(model, wav, out, batch=3) == 0


def test_build_map_matches_scientific_names(tmp_path):
    (tmp_path / "labels.csv").write_text("inat2024_fsd50k\nAra ararauna\nInsect\nPitangus sulphuratus\n")
    (tmp_path / "taxonomy.csv").write_text(
        "primary_label,inat_taxon_id,scientific_name,common_name,class_name\n"
        "grekis,16956,Pitangus sulphuratus,Great Kiskadee,Aves\n"
        "47158son01,47158,Insect son01,Insect sonotype01,Insecta\n"
        "baymac,19018,Ara ararauna,Blue-and-yellow Macaw,Aves\n"
    )
    m = build_map(tmp_path / "taxonomy.csv", tmp_path / "labels.csv")
    assert m.primary_label.tolist() == ["grekis", "baymac"]  # the sonotype has no Perch class
    assert m.perch_index.tolist() == [2, 0]


def test_prefetch_reads_only_a_few_files_ahead():
    started = []
    results = prefetch(lambda i: started.append(i) or i, range(100), ahead=4)
    assert next(results) == 0
    assert len(started) <= 5
    assert list(results) == list(range(1, 100))


def test_norm_stats(tmp_path):
    rng = np.random.default_rng(0)
    clips = {"a/1": rng.normal(0, 1, (300, 3)), "b/2": rng.normal(5, 1, (100, 3))}
    scape = rng.normal(-2, 1, (500, 3))
    for name, x in [*clips.items(), ("../train_soundscapes/s", scape)]:
        path = tmp_path / "train_audio" / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, logits=x.astype(np.float16))
    (tmp_path / "train.csv").write_text(
        "primary_label,secondary_labels,filename\na,['b'],a/1.ogg\nb,[],b/2.ogg\nnot_ours,[],a/9.ogg\n"
    )
    mean, top, missing = norm_stats(["a", "b", "c"], tmp_path, tmp_path / "train.csv")
    every = np.concatenate([*clips.values(), scape]).astype(np.float16).astype(np.float32)
    assert np.allclose(mean, every.mean(0), atol=1e-4)  # over clips and soundscapes
    a, b = (clips[k].astype(np.float16).astype(np.float32) for k in ("a/1", "b/2"))
    assert np.isclose(top[0], np.percentile(a[:, 0], 99), atol=1e-3)  # a: its primary clip only
    assert np.isclose(top[1], np.percentile(np.r_[a[:, 1], b[:, 1]], 99), atol=1e-3)  # b: primary and secondary
    assert missing.tolist() == [2]  # c has no clip: soundscape fallback
    assert np.isclose(top[2], np.percentile(scape.astype(np.float16).astype(np.float32)[:, 2], 99.9), atol=1e-3)


def test_online_embedding_matches_the_cached_model(tmp_path):
    import torch

    x = np.random.default_rng(0).uniform(-1, 1, (3, N)).astype(np.float32)
    path = fake_perch(tmp_path / "fake.onnx")
    online = PerchEmbedding(path, torch.device("cpu"))(torch.from_numpy(x))
    assert online.shape == (3, 1536) and np.allclose(online.numpy(), x[:, :1536])


def test_tensorrt_providers(tmp_path):
    import torch

    assert providers(torch.device("cpu"), "tensorrt") == ["CPUExecutionProvider"]
    trt, cuda = providers(torch.device("cuda:1"), "tensorrt", max_batch=32, cache=tmp_path / "trt")
    assert trt[0] == "TensorrtExecutionProvider" and cuda == ("CUDAExecutionProvider", {"device_id": 1})
    o = trt[1]
    assert o["device_id"] == 1 and o["trt_fp16_enable"] and o["trt_engine_cache_path"] == str(tmp_path / "trt" / "fp16")
    assert o["trt_profile_min_shapes"] == f"inputs:1x{N}" and o["trt_profile_max_shapes"] == f"inputs:32x{N}"
    fp32 = providers(torch.device("cuda:0"), "tensorrt-fp32", cache=tmp_path / "trt")[0][1]
    assert not fp32["trt_fp16_enable"] and fp32["trt_engine_cache_path"] == str(tmp_path / "trt" / "fp32")
    spec = providers(torch.device("cuda:0"), "tensorrt", 8, tmp_path, "spectrogram", (500, 128))[0][1]
    assert spec["trt_profile_max_shapes"] == "spectrogram:8x500x128"


def fake_perch_with_spectrogram(path):
    """inputs -> spectrogram (first 64000 samples as [B, 500, 128], times 2) -> embedding (its first 1536 values)."""
    ints = lambda name, v: helper.make_tensor(name, TensorProto.INT64, [len(v)], v)  # noqa: E731
    graph = helper.make_graph(
        [
            helper.make_node("Slice", ["inputs", "zero", "n_spec", "one"], ["head"]),
            helper.make_node("Reshape", ["head", "spec_shape"], ["grid"]),
            helper.make_node("Mul", ["grid", "two"], ["spectrogram"]),
            helper.make_node("Reshape", ["spectrogram", "flat_shape"], ["flat"]),
            helper.make_node("Slice", ["flat", "zero", "n_emb", "one"], ["embedding"]),
        ],
        "fake_perch_spec",
        [helper.make_tensor_value_info("inputs", TensorProto.FLOAT, ["batch", N])],
        [
            helper.make_tensor_value_info("spectrogram", TensorProto.FLOAT, ["batch", 500, 128]),
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 1536]),
        ],
        [
            ints("zero", [0]),
            ints("one", [1]),
            ints("n_spec", [64000]),
            ints("n_emb", [1536]),
            ints("spec_shape", [-1, 500, 128]),
            ints("flat_shape", [-1, 64000]),
            helper.make_tensor("two", TensorProto.FLOAT, [], [2.0]),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), path)
    return path


def test_split_matches_the_full_model(tmp_path):
    import torch

    path = fake_perch_with_spectrogram(tmp_path / "perch.onnx")
    front, net = split_perch(path)
    assert front == tmp_path / "perch" / "frontend.onnx" and net.exists()
    assert [i.name for i in onnx.load(net).graph.input] == ["spectrogram"]
    x = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1, (3, N)).astype(np.float32))
    full = PerchEmbedding(path, torch.device("cpu"))(x)
    split = PerchEmbedding(path, torch.device("cpu"), "split")(x)
    assert torch.allclose(full, split) and torch.allclose(full, 2 * x[:, :1536])


def test_split_recomputes_non_finite_rows_in_fp32(tmp_path):
    import torch

    class Overflowing:  # stands in for the FP16 engine: row 1 overflows
        def __init__(self, session):
            self.session = session

        def run(self, outputs, feeds):
            out = self.session.run(outputs, feeds)[0]
            out[1] = np.nan
            return [out]

    path = fake_perch_with_spectrogram(tmp_path / "perch.onnx")
    teacher = PerchEmbedding(path, torch.device("cpu"), "split")
    teacher.session = Overflowing(teacher.session)
    x = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1, (3, N)).astype(np.float32))
    e = teacher(x)
    assert torch.allclose(e, 2 * x[:, :1536]) and teacher.fallback_rows == 1 and teacher.rows == 3
