"""Run Perch v2 (ONNX) on every 5 s window of every WAV, with a 1 s stride, and cache the outputs.

One .npz per WAV, mirroring config.WAV_DIR: `embedding` [W, 1536] and `logits` [W, C], both float16. `logits` holds
Perch's raw logits for the C classes that are both ours and Perch's (matched by scientific name), in taxonomy.csv
order; perch_map.csv in the output folder lists them, one row per column. Window k covers seconds [k, k + 5); a file
shorter than 5 s is zero-padded to one window. Files already cached are skipped.

Memory stays bounded: a few WAVs are read ahead, and windows are copied into one fixed batch buffer. The run stops if
ONNX Runtime cannot use CUDA, unless --cpu is given.

    python -m birdclef.perch --subsets train_audio train_soundscapes --batch 64
    python -m birdclef.perch --benchmark --batch 32     # CUDA vs TensorRT FP16 / FP32 embeddings, online teacher
"""

import argparse
import ast
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import config as C


def build_map(taxonomy=C.RAW_DIR / "taxonomy.csv", labels=C.PERCH_LABELS):
    names = pd.read_csv(labels).iloc[:, 0].str.strip().str.lower()
    index = dict(zip(names, range(len(names))))
    tax = pd.read_csv(taxonomy)[["primary_label", "scientific_name", "class_name"]]
    tax["perch_index"] = tax.scientific_name.str.strip().str.lower().map(index)
    return tax.dropna(subset="perch_index").astype({"perch_index": int}).reset_index(drop=True)


def windows(x, sr=C.SR, seconds=C.WINDOW, stride=C.STRIDE):
    n = seconds * sr
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return np.lib.stride_tricks.sliding_window_view(x, n)[:: stride * sr]


def norm_stats(classes, perch_dir=C.PERCH_DIR, train_csv=C.RAW_DIR / "train.csv"):
    """Per label: the mean of the cached logits over every window (clips and soundscapes), and the 99th percentile over
    the windows of the clips where the label is primary or secondary (the 99.9th over all soundscape windows for a label
    without such a clip). Normalizing with (x - mean) / (top - mean) maps them to 0 and 1."""
    perch_dir, column = Path(perch_dir), {c: i for i, c in enumerate(classes)}
    total, count, positives = np.zeros(len(classes)), 0, [[] for _ in classes]
    train = pd.read_csv(train_csv).astype({"primary_label": str})
    for f, p, s in zip(train.filename, train.primary_label, train.secondary_labels):
        path = (perch_dir / "train_audio" / f).with_suffix(".npz")
        if path.exists():
            x = np.load(path)["logits"].astype(np.float32)
            total, count = total + x.sum(0), count + len(x)
            for c in {p, *ast.literal_eval(s)} & column.keys():
                positives[column[c]].append(x[:, column[c]])
    top = np.array([np.percentile(np.concatenate(v), 99) if v else np.nan for v in positives])
    missing, scapes = np.flatnonzero(np.isnan(top)), []
    for path in sorted((perch_dir / "train_soundscapes").glob("*.npz")):
        x = np.load(path)["logits"].astype(np.float32)
        total, count = total + x.sum(0), count + len(x)
        scapes.append(x[:, missing])
    if len(missing):
        top[missing] = np.percentile(np.concatenate(scapes), 99.9, axis=0)
    return total / count, top, missing


class Perch:
    def __init__(self, columns, path=C.PERCH_ONNX):
        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.providers = self.session.get_providers()  # what the session really uses, not what is installed
        self.columns = np.asarray(columns)  # Perch column of each class we keep

    def __call__(self, x):
        emb, label = self.session.run(["embedding", "label"], {"inputs": np.ascontiguousarray(x, np.float32)})
        return {"embedding": emb, "logits": label[:, self.columns]}


def load_tensorrt():
    """Load the TensorRT 10 libraries of the tensorrt-cu12-libs package, so ONNX Runtime finds them without
    LD_LIBRARY_PATH. Does nothing if the package is not installed."""
    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is None:
        return []
    folder = Path(spec.origin).parent
    names = ["libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"]  # libnvinfer first
    found = [folder / n for n in names if (folder / n).exists()]
    for lib in found:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    return found


BACKENDS = ("cuda", "tensorrt", "tensorrt-fp32", "split")
SPEC = ("spectrogram", (500, 128))  # Perch's log-magnitude spectrogram, the input of its CNN


def providers(device, backend="cuda", max_batch=64, cache=C.TRT_CACHE, input_name="inputs", input_shape=None):
    """ONNX Runtime providers. "tensorrt" (FP16) and "tensorrt-fp32": an engine for batches of 1..max_batch inputs,
    built once and cached; operations TensorRT cannot run fall back to CUDA."""
    if device.type != "cuda":
        return ["CPUExecutionProvider"]
    cuda = ("CUDAExecutionProvider", {"device_id": device.index or 0})
    if backend == "cuda":
        return [cuda]
    fp16 = backend == "tensorrt"
    shape = "x".join(map(str, input_shape or (C.WINDOW * C.SR,)))
    cache = Path(cache) / ("fp16" if fp16 else "fp32")
    trt = {
        "device_id": device.index or 0,
        "trt_fp16_enable": fp16,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache),
        "trt_timing_cache_enable": True,
        "trt_profile_min_shapes": f"{input_name}:1x{shape}",
        "trt_profile_opt_shapes": f"{input_name}:{max(1, max_batch // 2)}x{shape}",
        "trt_profile_max_shapes": f"{input_name}:{max_batch}x{shape}",
    }
    Path(cache).mkdir(parents=True, exist_ok=True)
    load_tensorrt()
    return [("TensorrtExecutionProvider", trt), cuda]


def split_perch(path, out_dir=None):
    """Cut Perch at its log spectrogram: frontend.onnx (inputs -> spectrogram) and network.onnx
    (spectrogram -> embedding), written next to the model once."""
    from onnx.utils import extract_model

    path = Path(path)
    out_dir = Path(out_dir) if out_dir else path.with_suffix("")  # models/perch_v2_no_dft/
    out_dir.mkdir(parents=True, exist_ok=True)
    front, net = out_dir / "frontend.onnx", out_dir / "network.onnx"
    if not front.exists():
        extract_model(str(path), str(front), ["inputs"], [SPEC[0]])
    if not net.exists():
        extract_model(str(path), str(net), [SPEC[0]], ["embedding"])
    return front, net


def run(session, device, inputs, output, shape):
    """One ONNX Runtime call; on CUDA, input and output stay on the GPU (IO binding)."""
    import torch

    name, x = inputs
    if device.type != "cuda":
        return torch.from_numpy(session.run([output], {name: x.cpu().numpy()})[0]).to(x.device)
    out = torch.empty(len(x), *shape, device=x.device)
    torch.cuda.synchronize(x.device)  # x is ready before ONNX Runtime reads it
    binding, index = session.io_binding(), x.device.index or 0
    binding.bind_input(name, "cuda", index, np.float32, tuple(x.shape), x.data_ptr())
    binding.bind_output(output, "cuda", index, np.float32, tuple(out.shape), out.data_ptr())
    session.run_with_iobinding(binding)
    return out


class PerchEmbedding:
    """Perch embedding of a [B, 160000] float tensor, computed on the tensor's device. Backends: "cuda",
    "tensorrt" (FP16), "tensorrt-fp32" (see providers), and "split": the log spectrogram on CUDA in FP32, the CNN
    on TensorRT FP16 (see split_perch); rows the FP16 CNN returns with a NaN or inf are recomputed by the same CNN in
    FP32 on CUDA (counted in fallback_rows)."""

    def __init__(self, path, device, backend="cuda", max_batch=64):
        import onnxruntime as ort

        self.device, self.backend = device, backend
        if backend == "split":
            front, net = split_perch(path)
            self.front = ort.InferenceSession(str(front), providers=providers(device, "cuda"))
            spec = providers(device, "tensorrt", max_batch, input_name=SPEC[0], input_shape=SPEC[1])
            self.session = ort.InferenceSession(str(net), providers=spec)
            self.fallback = ort.InferenceSession(str(net), providers=providers(device, "cuda"))
            self.rows = self.fallback_rows = 0
        else:
            self.session = ort.InferenceSession(str(path), providers=providers(device, backend, max_batch))
        wanted = "CUDAExecutionProvider" if backend == "cuda" else "TensorrtExecutionProvider"
        if device.type == "cuda" and wanted not in self.session.get_providers():
            raise RuntimeError(f"ONNX Runtime cannot use {wanted} for the online Perch teacher")

    def __call__(self, x):
        x = x.detach().float().contiguous()
        if self.backend == "split":
            spec = run(self.front, self.device, ("inputs", x), SPEC[0], SPEC[1])
            e = run(self.session, self.device, (SPEC[0], spec), "embedding", (1536,))
            bad = ~e.isfinite().all(1)
            n = int(bad.sum())
            self.rows, self.fallback_rows = self.rows + len(e), self.fallback_rows + n
            if n:
                e[bad] = run(self.fallback, self.device, (SPEC[0], spec[bad].contiguous()), "embedding", (1536,))
            return e
        return run(self.session, self.device, ("inputs", x), "embedding", (1536,))


def benchmark(onnx_path, wav_dir=C.WAV_DIR, n=512, batch=32, repeats=3, out=Path("results/perch_teacher.json")):
    """Perch embeddings of n real 5 s windows of soundscape WAVs with each backend: ms per batch, timed over full
    passes over all windows (so including the FP32 fallback of "split"), rows with a NaN or inf, and against CUDA FP32
    (on the rows finite in both) the largest absolute difference and smallest cosine."""
    import json
    import time

    import torch

    files = sorted((Path(wav_dir) / "train_soundscapes").glob("*.wav"))
    rng = np.random.default_rng(0)
    x = np.concatenate([windows(load(f))[:: C.WINDOW] for f in rng.choice(files, max(1, n // 12), replace=False)])[:n]
    x = torch.from_numpy(np.ascontiguousarray(x)).cuda()
    res, ref = {"windows": len(x), "batch": batch}, None
    for backend in BACKENDS:
        teacher = PerchEmbedding(onnx_path, x.device, backend, max_batch=batch)
        batches = [x[i : i + batch] for i in range(0, len(x), batch)]
        e = torch.cat([teacher(b) for b in batches])  # also builds the engine
        fallback = getattr(teacher, "fallback_rows", None)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            for b in batches:
                teacher(b)
        torch.cuda.synchronize()
        r = {"ms_per_batch": round((time.perf_counter() - start) / (repeats * len(batches)) * 1000, 2)}
        if fallback is not None:
            r["fallback_rows"] = fallback  # recomputed in FP32, so non_finite_rows counts what remains
        finite = torch.isfinite(e).all(1)
        r["non_finite_rows"] = int((~finite).sum())
        if ref is None:
            ref, ref_finite = e, finite
        else:
            ok = finite & ref_finite
            r["max_abs_diff"] = float((e[ok] - ref[ok]).abs().max())
            r["min_cosine"] = float(torch.nn.functional.cosine_similarity(e[ok], ref[ok], dim=1).min())
            r["speedup"] = round(res["cuda"]["ms_per_batch"] / r["ms_per_batch"], 2)
        res[backend] = r
        print(backend, r, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return res


def load(path):
    return sf.read(path, dtype="float32")[0]


def save(dst, arrays):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.npz")  # renamed only once fully written
    np.savez(tmp, **arrays)
    os.replace(tmp, dst)


def prefetch(fn, items, ahead):
    items = iter(items)
    with ThreadPoolExecutor(ahead) as ex:
        queue = deque(ex.submit(fn, x) for x in islice(items, ahead))
        while queue:
            result = queue.popleft().result()
            queue.extend(ex.submit(fn, x) for x in islice(items, 1))
            yield result


def cache_all(model, src_root=C.WAV_DIR, dst_root=C.PERCH_DIR, subsets=C.SUBSETS, batch=64, workers=4):
    src_root, dst_root = Path(src_root), Path(dst_root)
    src = [p for s in subsets for p in sorted((src_root / s).rglob("*.wav")) if not p.name.endswith(".part.wav")]
    dst = [dst_root / p.relative_to(src_root).with_suffix(".npz") for p in src]
    todo = [(s, d) for s, d in zip(src, dst) if not d.exists()]
    buf = np.empty((batch, C.WINDOW * C.SR), np.float32)
    segments, n, done = [], 0, 0  # segments: (file, rows of buf) in buffer order

    def run():
        nonlocal n, done
        out, a = model(buf[:n]), 0
        for f, c in segments:
            for k, v in out.items():
                f["parts"].setdefault(k, []).append(v[a : a + c].astype(np.float16))
            a, f["left"] = a + c, f["left"] - c
            if f["left"] == 0:
                save(f["dst"], {k: np.concatenate(v) for k, v in f["parts"].items()})
                done += 1
                if done % 500 == 0:
                    print(f"{done}/{len(todo)} files", flush=True)
        segments.clear()
        n = 0

    for d, x in prefetch(lambda t: (t[1], load(t[0])), todo, workers):
        w = windows(x)
        f, i = {"dst": d, "left": len(w), "parts": {}}, 0
        while i < len(w):
            c = min(batch - n, len(w) - i)
            buf[n : n + c] = w[i : i + c]
            segments.append((f, c))
            n, i = n + c, i + c
            if n == batch:
                run()
    if n:
        run()
    print(f"{done}/{len(todo)} files, {len(src) - len(todo)} already cached", flush=True)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=C.PERCH_ONNX)
    ap.add_argument("--labels", default=C.PERCH_LABELS)
    ap.add_argument("--taxonomy", default=C.RAW_DIR / "taxonomy.csv")
    ap.add_argument("--src", default=C.WAV_DIR)
    ap.add_argument("--dst", default=C.PERCH_DIR)
    ap.add_argument("--subsets", nargs="+", default=list(C.SUBSETS))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4, help="WAV files read ahead in background threads")
    ap.add_argument("--cpu", action="store_true", help="allow running without CUDA")
    ap.add_argument("--benchmark", action="store_true", help="compare CUDA and TensorRT embeddings, then exit")
    args = ap.parse_args()
    if args.benchmark:
        benchmark(args.onnx, args.src, batch=args.batch)
        raise SystemExit
    m = build_map(args.taxonomy, args.labels)
    model = Perch(m.perch_index, args.onnx)
    print(f"{len(m)} classes in both Perch and taxonomy.csv; ONNX Runtime providers: {model.providers}", flush=True)
    if "CUDAExecutionProvider" not in model.providers and not args.cpu:
        raise SystemExit("CUDA is not active in ONNX Runtime (see the warnings above); fix it or pass --cpu")
    Path(args.dst).mkdir(parents=True, exist_ok=True)
    m.to_csv(Path(args.dst) / "perch_map.csv", index=False)
    cache_all(model, args.src, args.dst, args.subsets, args.batch, args.workers)
