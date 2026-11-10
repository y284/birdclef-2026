"""Export trained models to ONNX and benchmark PyTorch, ONNX Runtime and OpenVINO on CPU, in FP32 and FP16.

The log-mel is computed once per soundscape in PyTorch (timed apart) and shared by every model; each backend runs the
CNNs on it. Each backend scores the held-out labelled soundscapes as a submission does: the whole recording -> one
frame per second -> max over each 5 s window -> smoothing with the neighbouring windows. Reported per backend: seconds
per 60 s soundscape for all the models (log-mel excluded), the max absolute difference of the averaged frame logits to
PyTorch FP32, and the macro ROC-AUC of the labelled segments, before (`auc_windows`) and after (`auc_smoothed`) the
smoothing. ONNX Runtime FP16 runs a converted FP16 graph; OpenVINO FP16 and BF16 compile the FP32 graph with that
inference precision, which falls back to FP32 on CPUs without native support (the report gives the precision used).
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn

from . import config as C
from .data import clip_items, load_classes, soundscape_segments, split
from .infer import submission_scores, to_windows
from .model import LogMel
from .pseudo import load_models
from .train import multi_hot
from .utils import macro_auc

SECONDS = 60


class Frames(nn.Module):
    """Log-mel [B, 1, mels, frames] -> frame logits [B, seconds, classes]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        return self.model.head(self.model.backbone(mel))[1]


def export_onnx(model, path, seconds=SECONDS):
    """FP32 graph at path, and its FP16 conversion (FP32 inputs and outputs) next to it."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    mel = torch.randn(1, 1, C.N_MELS, seconds * C.FRAMES_PER_SECOND)
    torch.onnx.export(
        Frames(model).eval(), (mel,), path, input_names=["mel"], output_names=["frames"], dynamo=True, verbose=False
    )
    onnx.save(convert_float_to_float16(onnx.load(path), keep_io_types=True), fp16_path(path))
    return path


def fp16_path(path):
    return Path(path).with_suffix(".fp16.onnx")


def torch_runner(models, threads):
    torch.set_num_threads(threads)
    nets = [Frames(m).eval() for m in models]

    @torch.no_grad()
    def run(mel):
        x = torch.from_numpy(mel)
        return np.mean([n(x).numpy() for n in nets], 0)

    return run


def onnxruntime_runner(paths, threads, fp16=False):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    paths = [fp16_path(p) if fp16 else p for p in paths]
    sessions = [ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"]) for p in paths]
    return lambda mel: np.mean([s.run(None, {"mel": mel})[0] for s in sessions], 0)


def openvino_runner(paths, threads, precision="f32"):
    import openvino as ov

    core = ov.Core()
    config = {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": precision}
    compiled = [core.compile_model(core.read_model(str(p)), "CPU", config) for p in paths]
    run = lambda mel: np.mean([c(mel)[0] for c in compiled], 0)  # noqa: E731
    run.precision = compiled[0].get_property("INFERENCE_PRECISION_HINT").get_type_name()
    return run


def cpu_name():
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        return next(line.split(":", 1)[1].strip() for line in lines if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


def load_wave(path, seconds=SECONDS):
    x = sf.read(path, dtype="float32")[0][: seconds * C.SR]
    return np.pad(x, (0, seconds * C.SR - len(x)))[None]


def held_out(raw, valid, split_seed):
    """The held-out labelled soundscape segments (as in train.py), their labels and the evaluated classes."""
    classes = load_classes(raw)
    clips = clip_items(classes, raw / "train.csv")
    segments = soundscape_segments(raw / "train_soundscapes_labels.csv")
    _, valid_scapes, evaluated = split(clips, segments, classes, valid, split_seed, raw / "train.csv")
    held = segments[segments.filename.isin(valid_scapes)].reset_index(drop=True)
    return held, multi_hot(held.labels, classes), evaluated, classes


def log_mels(waves):
    """Log-mel of every soundscape, and the mean seconds per soundscape."""
    mel, out, t = LogMel(), {}, time.perf_counter()
    with torch.no_grad():
        for name, wave in waves.items():
            out[name] = mel(torch.from_numpy(wave)).numpy()
    return out, (time.perf_counter() - t) / len(waves)


def benchmark(run, mels, repeats=1):
    """Frame logits of every soundscape and the mean seconds per soundscape (after one warm-up call)."""
    run(next(iter(mels.values())))
    frames, times = {}, []
    for name, mel in mels.items():
        t = time.perf_counter()
        for _ in range(repeats):
            frames[name] = run(mel)[0]
        times.append((time.perf_counter() - t) / repeats)
    return frames, float(np.mean(times))


def segment_scores(frames, held, smoothed):
    fn = submission_scores if smoothed else to_windows
    scores = {f: fn(torch.from_numpy(x)).numpy() for f, x in frames.items()}
    return np.stack([scores[f][s // C.WINDOW] for f, s in zip(held.filename, held.start)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", type=Path, required=True, help="checkpoints, frame logits averaged")
    p.add_argument("--backbone", default="tf_efficientnet_b0.ns_jft_in1k")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3, help="timed runs per soundscape")
    p.add_argument("--valid", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--raw", type=Path, default=C.RAW_DIR)
    p.add_argument("--wav", type=Path, default=C.WAV_DIR)
    p.add_argument("--onnx-dir", type=Path, default=Path("outputs/onnx"))
    p.add_argument("--out", type=Path, default=Path("results/export_benchmark.json"))
    args = p.parse_args()

    held, y, evaluated, classes = held_out(args.raw, args.valid, args.split_seed)
    models = load_models(args.models, len(classes), args.backbone, torch.device("cpu"))
    args.onnx_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        export_onnx(m, args.onnx_dir / f"{i}_{c.parent.name}_{c.stem}.onnx")
        for i, (m, c) in enumerate(zip(models, args.models))
    ]
    waves = {f: load_wave((args.wav / "train_soundscapes" / f).with_suffix(".wav")) for f in held.filename.unique()}
    torch.set_num_threads(args.threads)
    mels, mel_sec = log_mels(waves)

    runners = {
        "pytorch_fp32": lambda: torch_runner(models, args.threads),
        "onnxruntime_fp32": lambda: onnxruntime_runner(paths, args.threads),
        "onnxruntime_fp16": lambda: onnxruntime_runner(paths, args.threads, fp16=True),
        "openvino_fp32": lambda: openvino_runner(paths, args.threads, "f32"),
        "openvino_fp16": lambda: openvino_runner(paths, args.threads, "f16"),
        "openvino_bf16": lambda: openvino_runner(paths, args.threads, "bf16"),
    }
    report = {
        "models": [str(m) for m in args.models],
        "cpu": cpu_name(),
        "threads": args.threads,
        "soundscapes": len(waves),
        "segments": len(held),
        "classes_evaluated": int(evaluated.sum()),
        "log_mel_sec_per_soundscape": round(mel_sec, 4),
        "backends": {},
    }
    reference = None
    for name, make in runners.items():
        run = make()
        frames, sec = benchmark(run, mels, args.repeats)
        reference = reference or frames
        report["backends"][name] = {
            "sec_per_soundscape": round(sec, 4),
            "precision": getattr(run, "precision", name.split("_")[1].replace("fp", "f")),
            "max_abs_diff": float(max(np.abs(frames[f] - reference[f]).max() for f in frames)),
            "auc_windows": float(macro_auc(y[:, evaluated], segment_scores(frames, held, False)[:, evaluated])[0]),
            "auc_smoothed": float(macro_auc(y[:, evaluated], segment_scores(frames, held, True)[:, evaluated])[0]),
        }
        print(name, report["backends"][name], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(pd.DataFrame(report["backends"]).T.to_string())


if __name__ == "__main__":
    main()
