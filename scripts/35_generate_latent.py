"""Generate M futures per held-out cell from a latent model (34) and decode them to 128 px.

Context is the cell's 0 h frame encoded with the model's own copy of the autoencoder, the
fate the cell's own (two or three classes, as the model was trained, run.json "fates"); the
flow runs in the 4 x 32 x 32 code (frame model: each of frames 6, 8, 10, 12 from its own noise
draw; video model: the clip 6 to 12 jointly) and every code is decoded to a 128 px image.
Files are those of 19_generate_samples.py and 29_generate_clips.py, so the scoring chain
runs unchanged: samples[_start_stop].npy (cells, 4, M, 3, 128, 128) uint8 with its tracks
file, and for a video model also clips[_start_stop].npy (cells, M, 7, 3, 128, 128).

Usage: uv run scripts/35_generate_latent.py [model dir] [M, default 8] [cells: n or start:stop]
"""

import contextlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.flow import UNet, sample  # noqa: E402
from static2fate.latent import decode, encode, float_to_uint8, load_autoencoder  # noqa: E402
from static2fate.video import VideoUNet, sample_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FUTURE = [6, 8, 10, 12]
CELLS_PER_BATCH = {"cuda": 16, "mps": 4, "cpu": 2}


def main(mdir: Path, m: int, cells: str | None) -> None:
    train = importlib.import_module("09_train_flow")
    run = json.loads((mdir / "run.json").read_text())
    video = run["model"] == "video"
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.5)
    ae, std, _ = load_autoencoder(mdir / "ae", device)
    fates = int(run.get("fates", 2))
    model = (VideoUNet(ch=run["ch"], in_ch=run["z"], n_fates=fates) if video
             else UNet(ch=run["ch"], context=True, in_ch=run["z"], n_fates=fates))
    model.load_state_dict(torch.load(mdir / "model.pt", map_location=device, weights_only=True))
    model = model.to(device).eval()
    x, _, tracks = train.load("test", run["size"], frames=[0])
    fate = train.fate_classes(tracks, fates)
    suffix = ""
    if cells:
        a, _, b = cells.partition(":")
        sel = slice(int(a), int(b)) if b else slice(0, int(a))
        x, fate, tracks = x[sel], fate[sel], tracks[sel]
        suffix = f"_{sel.start}_{sel.stop}"
    n, size, ls = len(x), run["size"], run["latent_size"]
    clip = run["clip"] if video else FUTURE
    t = len(clip)
    print(f"cells {n}, {m} {'clips' if video else 'samples'} each, latent {run['z']}x{ls}x{ls}, "
          f"decoded to {size} px, {fates} fate classes, device {device}", flush=True)
    amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()
    z0 = encode(ae, x[:, 0], std)  # (n, z, ls, ls) context codes

    samples = np.lib.format.open_memmap(mdir / f"samples{suffix}.npy", mode="w+", dtype=np.uint8,
                                        shape=(n, len(FUTURE), m, 3, size, size))
    clips = None
    if video:
        clips = np.lib.format.open_memmap(mdir / f"clips{suffix}.npy", mode="w+", dtype=np.uint8,
                                          shape=(n, m, t, 3, size, size))
        keep = [clip.index(f) for f in FUTURE]
    g = torch.Generator().manual_seed(0)
    per_batch = CELLS_PER_BATCH[device]
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
                out = sample_video(model, frame, fk, ctx, cf, x0=noise, clamp=False)
            u8 = float_to_uint8(decode(ae, out.flatten(0, 1).float(), std)).view(k, m, t, 3, size, size)
            clips[sl] = u8.numpy()
            samples[sl] = u8[:, :, keep].transpose(1, 2).numpy()
        else:
            for j, f in enumerate(FUTURE):
                noise = torch.randn(k * m, run["z"], ls, ls, generator=g).to(device)
                with amp:
                    out = sample(model, torch.full((k * m,), f, device=device), fk, x0=noise, clamp=False,
                                 ctx=ctx, ctx_frame=cf)
                samples[sl, j] = float_to_uint8(decode(ae, out.float(), std)).view(k, m, 3, size, size).numpy()
        if device == "mps":
            torch.mps.empty_cache()
        if sl.stop % 16 == 0 or sl.stop == n:
            print(f"cells {sl.stop}/{n}", flush=True)
    samples.flush()
    np.save(mdir / f"samples{suffix}_tracks.npy", tracks)
    if video:
        clips.flush()
        np.save(mdir / f"clips{suffix}_tracks.npy", tracks)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/latent_frame",
         int(sys.argv[2]) if len(sys.argv) > 2 else 8,
         sys.argv[3] if len(sys.argv) > 3 else None)
