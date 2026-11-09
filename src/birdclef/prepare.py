"""Decode the competition OGG files once into mono 16-bit WAV at config.SR, mirroring the folder layout.

Files already converted are skipped, so an interrupted run can be restarted. With --delete-src, each OGG is deleted
once its WAV is complete (also for WAVs converted by an earlier run).

    python -m birdclef.prepare --subsets train_audio train_soundscapes --workers 8 [--delete-src]
"""

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import config as C


def load(path, sr=C.SR):
    x, rate = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if rate != sr:
        g = gcd(rate, sr)
        x = resample_poly(x, sr // g, rate // g).astype(np.float32)
    return x


def convert(src, dst, delete_src=False):
    done = 0
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part.wav")  # renamed only once fully written
        sf.write(tmp, np.clip(load(src), -1, 1), C.SR, subtype="PCM_16")
        os.replace(tmp, dst)
        done = 1
    if delete_src:
        src.unlink()
    return done


def convert_all(src_root=C.RAW_DIR, dst_root=C.WAV_DIR, subsets=C.SUBSETS, workers=None, delete_src=False):
    src_root, dst_root = Path(src_root), Path(dst_root)
    src = [p for s in subsets for p in sorted((src_root / s).rglob("*.ogg"))]
    dst = [dst_root / p.relative_to(src_root).with_suffix(".wav") for p in src]
    done = 0
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for i, n in enumerate(ex.map(convert, src, dst, [delete_src] * len(src), chunksize=32), 1):
            done += n
            if i % 1000 == 0 or i == len(src):
                print(f"{i}/{len(src)} files, {done} converted", flush=True)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=C.RAW_DIR)
    ap.add_argument("--dst", default=C.WAV_DIR)
    ap.add_argument("--subsets", nargs="+", default=list(C.SUBSETS))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--delete-src", action="store_true", help="delete each OGG once its WAV is complete")
    args = ap.parse_args()
    convert_all(args.src, args.dst, args.subsets, args.workers, args.delete_src)
