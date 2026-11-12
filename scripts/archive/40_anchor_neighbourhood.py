"""Keep the context's neighbourhood, draw the cell: anchored sampling in the latent.

The latent models get the cell and invent its surroundings; the pixel model keeps the
surroundings and blurs the cell (docs/log.md, "Where the DINOv2 distance lives"). Here the
latent sampler is held, outside a disc around the crop centre, on the straight path from its
own noise to the context's code: after every Euler step the state outside the disc is reset to
(1 - s) * noise + s * z_context, inside it the model is free. At s = 1 the outside is the
context's code exactly, so the decoded neighbourhood is the 0 h neighbourhood (as the pixel
model in effect keeps it) and only the cell is generated, conditioned on the whole context and
seeing the fixed surroundings at every step. Radii are given at 128 px (the disc of
39_energy_masked.py: 40 px is the central cell and little else) and scaled to the 32 px code;
the edge is softened over one code pixel. Output in the layout of 35_generate_latent.py, one
folder per model and radius, so the scoring chain runs with name=dir arguments.

Usage: uv run scripts/archive/40_anchor_neighbourhood.py [model dir] [M, default 8] [radii, default 40,56] [cells: n or start:stop]
Writes results/anchor/<model dir name>_r<radius>/samples.npy (+ clips.npy for a video model),
samples_tracks.npy and run.json.
"""

import contextlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))  # scripts/, for the current pipeline
from static2fate.flow import UNet, sample  # noqa: E402
from static2fate.latent import decode, encode, float_to_uint8, load_autoencoder  # noqa: E402
from static2fate.video import VideoUNet, sample_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/anchor"
FUTURE = [6, 8, 10, 12]
CELLS_PER_BATCH = {"cuda": 16, "mps": 4, "cpu": 2}


def disc_weights(size: int, radius: float, edge: float = 1.0) -> torch.Tensor:
    """(size, size): 1 inside the disc (free), 0 outside (anchored), a linear ramp of `edge` px."""
    c = (size - 1) / 2
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    r = torch.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    return ((radius - r) / edge + 0.5).clamp(0, 1)


def anchored(noise: torch.Tensor, target: torch.Tensor, free: torch.Tensor):
    """project(x, s): x inside the disc, (1 - s) * noise + s * target outside it (broadcast over x)."""
    def project(x: torch.Tensor, s: float) -> torch.Tensor:
        return free * x + (1 - free) * ((1 - s) * noise + s * target)
    return project


def main(mdir: Path, m: int, radii: list[float], cells: str | None) -> None:
    train = importlib.import_module("09_train_flow")
    mdir = mdir.resolve()
    run = json.loads((mdir / "run.json").read_text())
    video = run["model"] == "video"
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.5)
    ae, std, _ = load_autoencoder(mdir / "ae", device)
    model = (VideoUNet(ch=run["ch"], in_ch=run["z"]) if video else UNet(ch=run["ch"], context=True, in_ch=run["z"]))
    model.load_state_dict(torch.load(mdir / "model.pt", map_location=device, weights_only=True))
    model = model.to(device).eval()
    x, fate, tracks = train.load("test", run["size"], frames=[0])
    suffix = ""
    if cells:
        a, _, b = cells.partition(":")
        sel = slice(int(a), int(b)) if b else slice(0, int(a))
        x, fate, tracks = x[sel], fate[sel], tracks[sel]
        suffix = f"_{sel.start}_{sel.stop}"
    n, size, ls = len(x), run["size"], run["latent_size"]
    clip = run["clip"] if video else FUTURE
    t = len(clip)
    amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()
    z0 = encode(ae, x[:, 0], std)  # (n, z, ls, ls) context codes
    per_batch = CELLS_PER_BATCH[device]
    for radius in radii:
        out_dir = OUT / f"{mdir.name}_r{radius:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        free = disc_weights(ls, radius * ls / size).to(device)
        print(f"{out_dir.name}: cells {n}, {m} {'clips' if video else 'samples'} each, disc radius {radius} px "
              f"({radius * ls / size:.1f} code px, {free.mean():.0%} of the code free), device {device}", flush=True)
        samples = np.lib.format.open_memmap(out_dir / f"samples{suffix}.npy", mode="w+", dtype=np.uint8,
                                            shape=(n, len(FUTURE), m, 3, size, size))
        clips = None
        if video:
            clips = np.lib.format.open_memmap(out_dir / f"clips{suffix}.npy", mode="w+", dtype=np.uint8,
                                              shape=(n, m, t, 3, size, size))
            keep = [clip.index(f) for f in FUTURE]
        g = torch.Generator().manual_seed(0)  # the same noise as 35_generate_latent.py
        t0 = time.time()
        for b in range(0, n, per_batch):
            sl = slice(b, min(b + per_batch, n))
            k = sl.stop - sl.start
            ctx = z0[sl].repeat_interleave(m, 0).to(device)
            fk = fate[sl].repeat_interleave(m).to(device)
            cf = torch.zeros(k * m, dtype=torch.long, device=device)
            if video:
                frame = torch.tensor(clip, device=device)[None].expand(k * m, -1)
                noise = torch.randn(k * m, t, run["z"], ls, ls, generator=g).to(device)
                with amp:
                    out = sample_video(model, frame, fk, ctx, cf, x0=noise, clamp=False,
                                       project=anchored(noise, ctx[:, None], free))
                u8 = float_to_uint8(decode(ae, out.flatten(0, 1).float(), std)).view(k, m, t, 3, size, size)
                clips[sl] = u8.numpy()
                samples[sl] = u8[:, :, keep].transpose(1, 2).numpy()
            else:
                for j, f in enumerate(FUTURE):
                    noise = torch.randn(k * m, run["z"], ls, ls, generator=g).to(device)
                    with amp:
                        out = sample(model, torch.full((k * m,), f, device=device), fk, x0=noise, clamp=False,
                                     project=anchored(noise, ctx, free), ctx=ctx, ctx_frame=cf)
                    samples[sl, j] = float_to_uint8(decode(ae, out.float(), std)).view(k, m, 3, size, size).numpy()
            if device == "mps":
                torch.mps.empty_cache()
            if sl.stop % 16 == 0 or sl.stop == n:
                print(f"  cells {sl.stop}/{n}, {(time.time() - t0) / 60:.1f} min", flush=True)
        samples.flush()
        np.save(out_dir / f"samples{suffix}_tracks.npy", tracks)
        if video:
            clips.flush()
            np.save(out_dir / f"clips{suffix}_tracks.npy", tracks)
        (out_dir / "run.json").write_text(json.dumps({
            "model_dir": str(mdir.relative_to(ROOT) if mdir.is_relative_to(ROOT) else mdir), "model": run["model"], "radius_px": radius,
            "radius_code_px": radius * ls / size, "edge_code_px": 1.0, "free_fraction": float(free.mean()),
            "m": m, "cells": n, "seed": 0, "device": device, "minutes": (time.time() - t0) / 60}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/kaggle/lat_video",
         int(sys.argv[2]) if len(sys.argv) > 2 else 8,
         [float(r) for r in sys.argv[3].split(",")] if len(sys.argv) > 3 else [40.0, 56.0],
         sys.argv[4] if len(sys.argv) > 4 else None)
