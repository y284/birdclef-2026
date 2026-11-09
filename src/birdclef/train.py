"""Train one model per seed on the training clips and labelled soundscapes, with an EMA of the weights and a masked
cross-entropy on the clip logits; with --distill w, plus w * (1 - cosine similarity) between the projected backbone
features and Perch's embedding of the crop. With --pseudo-dir, every crop is mixed with a pseudo-labelled soundscape
crop (data.mix_pseudo); with --mixup p, a fraction p of each batch is then mixed in pairs (data.mixup). With --distill,
Perch's embedding of every mixed sample is computed online on the GPU. Every epoch, the EMA model is scored
on a random held-out split (clips and soundscapes, labels with more than config.RARE clips) and the best epoch is
kept as model_<run>_seed<seed>_best.pt, <run> starting with the backbone (e.g. tf_efficientnet_b0_perch_pseudo).

    python -m birdclef.train --selection random --seeds 42
    python -m birdclef.train --selection perch --pseudo --seeds 42
    python -m birdclef.train --selection perch --pseudo --distill 1 --seeds 42
    python -m birdclef.train --selection perch --pseudo --distill 1 --mixup 0.5 --seeds 42
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import ConcatDataset, DataLoader

from . import config as C
from .data import (
    ClipDataset,
    PseudoSoundscapes,
    SoundscapeDataset,
    WithPseudo,
    clip_items,
    load_classes,
    mix_pseudo,
    mixup,
    soundscape_items,
    soundscape_segments,
    split,
)
from .infer import evaluate
from .model import Net, cosine_loss, masked_ce
from .perch import BACKENDS, PerchEmbedding
from .utils import amp_dtype, autocast, run_on_devices, seed_everything


def multi_hot(groups, classes):
    index = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(groups), len(classes)), np.int8)
    for i, labels in enumerate(groups):
        y[i, [index[c] for c in labels if c in index]] = 1
    return y


def train(device, job, data, valid, args):
    seed, model = job
    tag = f"{args.run} seed {seed} · {device}"
    model.to(device).train()
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    loader = DataLoader(
        data,
        args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(loader), pct_start=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and amp_dtype(device) == torch.float16)
    history, best, start, eval_time, n_evals = [], -1.0, time.time(), 0.0, 0
    evaluated = [e for e in range(1, args.epochs + 1) if e % args.eval_every == 0 or e == args.epochs]
    teacher = None
    if (args.mixup or args.pseudo_dir) and args.distill:
        teacher = PerchEmbedding(args.perch_onnx, device, args.teacher_backend, max_batch=args.batch_size)
    generator = torch.Generator(device).manual_seed(seed)

    for epoch in range(1, args.epochs + 1):
        total = total_cos = 0.0
        epoch_start = time.time()
        for step, b in enumerate(loader, 1):
            wave = b["wave"].to(device, non_blocking=True)
            target, mask = b["target"].to(device), b["mask"].to(device)
            if args.distill:
                emb_target, has = b["embedding"].to(device), b["has_embedding"].to(device)
            holdout = b["holdout"].to(device) > 0
            changed = torch.zeros_like(holdout)  # samples whose audio no longer matches the cached embedding
            if args.pseudo_dir:  # every crop mixed with a pseudo-labelled soundscape crop
                wave, target = mix_pseudo(
                    wave, target, mask, b["pseudo_wave"].to(device), b["pseudo_target"].to(device), generator
                )
                changed = ~changed
            if args.mixup:  # with --pseudo-dir, a second mix of the already mixed crops
                wave, target, mask, mixed, perm = mixup(wave, target, mask, args.mixup, generator)
                holdout = holdout | (mixed & holdout[perm])  # a mix containing held-out audio
                changed = changed | mixed
            todo = changed & ~holdout
            if teacher is not None and todo.any():
                emb_target, has = emb_target.clone(), has.clone()
                emb_target[todo], has[todo] = teacher(wave[todo]), 1.0
            if args.distill:
                has = has * ~holdout  # no Perch target for held-out audio
            with autocast(device):
                if args.distill:
                    logits, _, embedding = model.forward_with_embedding(wave)
                else:
                    logits, _ = model(wave)
            loss = masked_ce(logits, target, mask)
            if args.distill:
                cos = cosine_loss(embedding, emb_target, has)
                loss = loss + args.distill * cos
                total_cos += cos.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update_parameters(model)
            total += loss.item()
            if step % args.log_every == 0 or step == len(loader):
                done, elapsed = (epoch - 1) * len(loader) + step, time.time() - start
                per_eval = eval_time / n_evals if n_evals else 0.0  # unknown until the first evaluation
                left = (elapsed - eval_time) / done * (args.epochs * len(loader) - done)
                left += per_eval * sum(e >= epoch for e in evaluated)
                print(
                    f"[{tag}] epoch {epoch}/{args.epochs} step {step}/{len(loader)} loss {total / step:.4f}"
                    + (f" (cosine {total_cos / step:.4f})" if args.distill else "")
                    + " | "
                    f"{elapsed / 60:.0f} min elapsed, ~{left / 60:.0f} min left",
                    flush=True,
                )

        t = time.time()
        history.append({"seed": seed, "epoch": epoch, "loss": round(total / len(loader), 4)})
        history[-1]["train_sec"] = round(t - epoch_start, 1)
        if args.distill:
            history[-1]["cosine_loss"] = round(total_cos / len(loader), 4)
        line = f"[{tag}] epoch {epoch} loss {history[-1]['loss']:.4f} | train {history[-1]['train_sec'] / 60:.1f} min"
        if epoch in evaluated:
            res = evaluate(ema.module, valid, device, args)
            eval_time, n_evals = eval_time + time.time() - t, n_evals + 1
            history[-1].update({k: round(v, 4) for k, v in res.items()})
            history[-1]["eval_sec"] = round(time.time() - t, 1)
            if res["score"] > best:
                best = res["score"]
                torch.save(ema.module.state_dict(), args.out / f"model_{args.run}_seed{seed}_best.pt")
            line += (
                f", eval {history[-1]['eval_sec'] / 60:.1f} min | clip AUC {res['clip_auc']:.4f} "
                f"soundscape AUC {res['soundscape_auc']:.4f}"
            )
        print(line, flush=True)
        pd.DataFrame(history).to_csv(args.out / f"history_{args.run}_seed{seed}.csv", index=False)

    torch.save(ema.module.state_dict(), args.out / f"model_{args.run}_seed{seed}.pt")
    top = max((h for h in history if "score" in h), key=lambda h: h["score"])
    later = [h["train_sec"] for h in history[1:]] or [history[0]["train_sec"]]  # epoch 1 includes warm-up
    summary = {"seed": seed, "best_epoch": top["epoch"], "train_min": round((time.time() - start) / 60, 1)}
    summary["train_sec_per_epoch"] = round(sum(later) / len(later), 1)
    summary.update({k: top[k] for k in ("clip_auc", "soundscape_auc", "score")})
    return summary, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", default="random", choices=["random", "perch"], help="crop centre in training clips")
    p.add_argument("--pseudo", action="store_true", help="soft targets from trusted Perch curves instead of 0")
    p.add_argument("--distill", type=float, default=0.0, help="weight of the cosine loss to Perch's embedding")
    p.add_argument("--mixup", type=float, default=0.0, help="fraction of each batch mixed with another crop")
    p.add_argument("--perch-onnx", type=Path, default=C.PERCH_ONNX, help="online teacher for mixed crops")
    p.add_argument("--teacher-backend", default="cuda", choices=list(BACKENDS))
    p.add_argument("--pseudo-dir", type=Path, help="mix every crop with a soundscape crop pseudo-labelled by pseudo.py")
    p.add_argument("--pseudo-pow", type=float, default=1.0, help="power of the softmaxed pseudo target")
    p.add_argument("--drop-path", type=float, default=0.0, help="drop path rate of the backbone")
    p.add_argument("--init", help="checkpoint to start from, {seed} replaced by the seed")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--backbone", default="tf_efficientnet_b0.ns_jft_in1k")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--valid", type=float, default=0.2, help="held-out fraction of clips and of labelled soundscapes")
    p.add_argument(
        "--holdout-unscored", action="store_true", help="also train on held-out soundscapes, for their unscored labels"
    )
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=24, help="5 s windows scored per held-out clip")
    p.add_argument("--eval-every", type=int, default=1, help="evaluate every N epochs (and always the last one)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--raw", type=Path, default=C.RAW_DIR)
    p.add_argument("--wav", type=Path, default=C.WAV_DIR)
    p.add_argument("--curves", type=Path, default=C.CURVES_DIR)
    p.add_argument("--perch", type=Path, default=C.PERCH_DIR)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--smoke", action="store_true", help="2 epochs on 64 training items and 16 held-out clips")
    args = p.parse_args()
    args.run = args.backbone.split(".")[0] + "_" + args.selection  # timm architecture, without the weights tag
    args.run += "_pseudo" * args.pseudo + "_distill" * (args.distill > 0) + "_mixup" * (args.mixup > 0)
    args.run += "_unscored" * args.holdout_unscored
    if args.pseudo_dir:
        args.run += f"_pl-{args.pseudo_dir.name}"  # e.g. _pl-iter1
    if (args.mixup or args.pseudo_dir) and args.distill:
        args.run += f"_{args.teacher_backend}"  # the online teacher
    args.out.mkdir(parents=True, exist_ok=True)

    classes = load_classes(args.raw)
    clips = clip_items(classes, args.raw / "train.csv")
    segments = soundscape_segments(args.raw / "train_soundscapes_labels.csv")
    valid_clips, valid_scapes, evaluated = split(
        clips, segments, classes, args.valid, args.split_seed, args.raw / "train.csv"
    )
    train_clips = clips[~clips.filename.isin(valid_clips)]
    train_scapes = soundscape_items(segments[~segments.filename.isin(valid_scapes)])
    held_items = soundscape_items(segments[segments.filename.isin(valid_scapes)])
    held_items = held_items[~held_items.label.map(dict(zip(classes, evaluated)))]  # unscored labels only
    kw = {"pseudo": args.pseudo, "embedding": args.distill > 0, "wav_dir": args.wav, "curves_dir": args.curves}
    kw["perch_dir"] = args.perch
    parts = [
        ClipDataset(train_clips, classes, args.selection, **kw),
        SoundscapeDataset(train_scapes, segments, classes, **kw),
    ]
    if args.holdout_unscored:
        parts.append(SoundscapeDataset(held_items, segments, classes, holdout=True, scored=evaluated, **kw))
    data = ConcatDataset(parts)
    if args.pseudo_dir:
        data = WithPseudo(data, PseudoSoundscapes(args.pseudo_dir, args.pseudo_pow, args.wav))
    held = clips[clips.filename.isin(valid_clips)].groupby("filename", sort=False).label.agg(list)
    held_segments = segments[segments.filename.isin(valid_scapes)].reset_index(drop=True)
    if args.smoke:
        args.epochs, held, args.max_windows = 2, held.iloc[:16], 2
        data = torch.utils.data.Subset(data, np.random.default_rng(0).choice(len(data), 64, replace=False))
    valid = {
        "clips": held.index.tolist(),
        "clip_y": multi_hot(held, classes),
        "segments": held_segments,
        "segment_y": multi_hot(held_segments.labels, classes),
        "evaluated": evaluated,
    }
    split_info = {
        "train_items": len(data),
        "valid_clips": len(held),
        "valid_soundscapes": len(valid_scapes),
        "valid_segments": len(held_segments),
        "holdout_unscored_items": len(held_items) * args.holdout_unscored,
        "pseudo_soundscapes": len(data.pseudo) if args.pseudo_dir else 0,
        "evaluated_classes": int(evaluated.sum()),
    }
    print(split_info, flush=True)

    jobs = []
    for seed in args.seeds:
        seed_everything(seed)
        model = Net(len(classes), args.backbone, pretrained=not args.no_pretrained, drop_path=args.drop_path)
        if args.init:
            path = args.init.format(seed=seed)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            print(f"seed {seed}: starting from {path}", flush=True)
        jobs.append((seed, model))
    results = run_on_devices(jobs, lambda device, job: train(device, job, data, valid, args))

    summary, history = [s for s, _ in results], [h for _, hs in results for h in hs]
    name = f"{args.run}_seed{'-'.join(map(str, args.seeds))}"  # a later run with other seeds writes its own files
    pd.DataFrame(history).to_csv(args.out / f"history_{name}.csv", index=False)
    (args.out / f"results_training_{name}.json").write_text(
        json.dumps(
            {
                "run": args.run,
                "args": vars(args) | {"out": str(args.out)},
                "split": split_info,
                "summary": summary,
                "history": history,
            },
            indent=2,
            default=str,
        )
    )
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
