"""Train the autoencoder (static2fate.autoencoder) on every frame of every training cell.

Images: all 13 frames of the training cells at 128 px, all ten doses by default (5,860 cells,
76k images; --two-fate restricts to the four doses the flow models use), held-out wells as
everywhere else. Random flips and 90 degree rotations. Loss: L1 + perceptual (VGG16) +
kl x KL, and from step --gan-start on, if --gan > 0, a hinge-loss PatchGAN term with that
fixed weight (discriminator trained in the same step on the detached reconstruction).

Usage: uv run scripts/31_train_autoencoder.py --minutes 60 [--device mps|cuda] [--batch 32]
       [--ch 64] [--z 4] [--kl 1e-6] [--perceptual 1.0] [--gan 0] [--gan-start 5000]
       [--two-fate] [--snapshot-every 120] [--out results/autoencoder]
Writes <out>/model.pt, <out>/disc.pt (if --gan), <out>/train_log.csv and <out>/run.json,
whose "latent_std" is the scale the flow models divide the latent by. Float16 on CUDA.
"""

import argparse
import contextlib
import importlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.autoencoder import AutoEncoder, Discriminator, hinge_d_loss, kl_loss, latent_std  # noqa: E402
from static2fate.perceptual import Perceptual  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--z", type=int, default=4)
    ap.add_argument("--kl", type=float, default=1e-6)
    ap.add_argument("--perceptual", type=float, default=1.0)
    ap.add_argument("--gan", type=float, default=0.0, help="weight of the adversarial term, 0 = none")
    ap.add_argument("--gan-start", type=int, default=5000, help="step the discriminator starts at")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=float, default=15)
    ap.add_argument("--snapshot-every", type=float, default=0, help="minutes between kept snapshots")
    ap.add_argument("--two-fate", action="store_true", help="only the four two-fate doses")
    ap.add_argument("--out", default=str(ROOT / "results/autoencoder"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = importlib.import_module("09_train_flow")
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(0.5)

    x8, _ = train.load_uint8("train", train.DOSES if args.two_fate else None)
    n, nf = x8.shape[:2]
    print(f"train cells {n}, images {n * nf}, size {args.size}", flush=True)
    ae = AutoEncoder(ch=args.ch, z=args.z).to(args.device)
    disc = Discriminator().to(args.device) if args.gan > 0 else None
    perceptual = Perceptual().to(args.device)
    print(f"autoencoder {sum(p.numel() for p in ae.parameters()) / 1e6:.1f} M parameters, "
          f"latent {args.z} x {args.size // ae.factor} x {args.size // ae.factor}", flush=True)
    opt = torch.optim.Adam(ae.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.9)) if disc else None
    amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=amp)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else contextlib.nullcontext

    def batch():
        ci = torch.randint(n, (args.batch,))
        fi = torch.randint(nf, (args.batch,))
        xb = x8[ci, fi].float().div_(127.5).sub_(1)
        if args.size != 128:
            xb = torch.nn.functional.avg_pool2d(xb, 128 // args.size)
        return train.augment(xb).to(args.device)

    def save(step, t0, final=False):
        torch.save(ae.state_dict(), out / "model.pt")
        if disc:
            torch.save(disc.state_dict(), out / "disc.pt")
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        info = {"device": args.device, "seed": args.seed, "steps": step, "batch": args.batch,
                "size": args.size, "ch": args.ch, "z": args.z, "factor": ae.factor, "kl": args.kl,
                "perceptual": args.perceptual, "gan": args.gan, "gan_start": args.gan_start,
                "lr": args.lr, "float16": amp, "two_fate": args.two_fate,
                "minutes": round((time.time() - t0) / 60, 1), "train_cells": n,
                "heldout_wells": train.HELDOUT_WELLS, "torch": torch.__version__}
        if final:
            ae.eval()
            sub = x8[torch.randperm(n)[:256]][:, ::4].flatten(0, 1).float().div_(127.5).sub_(1)
            if args.size != 128:
                sub = torch.nn.functional.avg_pool2d(sub, 128 // args.size)
            info["latent_std"] = latent_std(ae, sub)
            ae.train()
        (out / "run.json").write_text(json.dumps(info, indent=1))

    log, step, t0, last_save, last_snap = [], 0, time.time(), time.time(), time.time()
    while time.time() - t0 < args.minutes * 60:
        xb = batch()
        use_gan = disc is not None and step >= args.gan_start
        with autocast():
            recon, mean, logvar = ae(xb)
            rec = (recon - xb).abs().mean()
            per = perceptual(recon, xb).mean()
            kl = kl_loss(mean.float(), logvar.float())
            loss = rec + args.perceptual * per + args.kl * kl
            g = torch.zeros((), device=xb.device)
            if use_gan:
                g = -disc(recon).mean()
                loss = loss + args.gan * g
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        d = torch.zeros((), device=xb.device)
        if use_gan:
            with autocast():
                d = hinge_d_loss(disc(xb), disc(recon.detach()))
            opt_d.zero_grad()
            scaler_d.scale(d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        if step % 100 == 0:
            log.append({"step": step, "minutes": round((time.time() - t0) / 60, 2), "loss": loss.item(),
                        "l1": rec.item(), "perceptual": per.item(), "kl": kl.item(), "g": g.item(), "d": d.item()})
            print(log[-1], flush=True)
        if args.snapshot_every and time.time() - last_snap > args.snapshot_every * 60:
            torch.save(ae.state_dict(), out / f"model_{step}.pt")
            last_snap = time.time()
        if time.time() - last_save > args.save_every * 60:
            save(step, t0)
            last_save = time.time()
        step += 1

    save(step, t0, final=True)
    print(f"done, {step} steps", flush=True)


if __name__ == "__main__":
    main()
