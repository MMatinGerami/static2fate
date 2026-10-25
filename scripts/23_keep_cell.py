"""Futures that start from the cell itself: partial noise on the context image (SDEdit-style).

The context model sampled from pure noise makes a plausible cell of the right fate and time
but loses the cell (22_identity.py). Here the integration starts at s0 along the path from
noise to image, from (1 - s0) * noise + s0 * context image, so the large-scale layout of the
real cell survives, more for larger s0, and the model only has to fill in the change the fate
and the later frame ask for. One strength per output folder, same layout as
19_generate_samples.py, so 20, 21 and 22 can score them with name=dir arguments.

Usage: uv run scripts/23_keep_cell.py [model dir] [M, default 4] [s0 values, default 0.3,0.5,0.7]
Writes results/keep_cell/<model dir name>_s<s0>/samples.npy, samples_tracks.npy, run.json.
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

ROOT = Path(__file__).resolve().parents[1]
FUTURE = [6, 8, 10, 12]
CELLS_PER_BATCH = {64: 64, 128: 16}


def main(mdir: Path, m: int, strengths: list[float]) -> None:
    train = importlib.import_module("09_train_flow")
    run = json.loads((mdir / "run.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(ch=run["ch"], context=True).to(device)
    model.load_state_dict(torch.load(mdir / "model.pt", map_location=device, weights_only=True))
    model.eval()
    x, fate, tracks = train.load("test", run["size"], [0])
    n, size = len(x), run["size"]
    amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()
    per_batch = CELLS_PER_BATCH[size]
    for s0 in strengths:
        out_dir = ROOT / "results/keep_cell" / f"{mdir.name}_s{s0:.1f}".replace(".", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"s0 {s0}: cells {n}, {m} samples each, size {size}, device {device} -> {out_dir}", flush=True)
        out = np.lib.format.open_memmap(out_dir / "samples.npy", mode="w+", dtype=np.uint8,
                                        shape=(n, len(FUTURE), m, 3, size, size))
        g = torch.Generator().manual_seed(0)
        for b in range(0, n, per_batch):
            sl = slice(b, min(b + per_batch, n))
            k = sl.stop - sl.start
            ctx = x[sl, 0].repeat_interleave(m, 0).to(device)
            fk = fate[sl].repeat_interleave(m).to(device)
            for j, f in enumerate(FUTURE):
                z = torch.randn(k * m, 3, size, size, generator=g)
                with amp:
                    im = sample(model, torch.full((k * m,), f, device=device), fk, x0=z.to(device),
                                x_start=ctx, s0=s0, ctx=ctx,
                                ctx_frame=torch.zeros(k * m, dtype=torch.long, device=device)).float().cpu()
                out[sl, j] = ((im + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).view(k, m, 3, size, size).numpy()
            print(f"  cells {sl.stop}/{n}", flush=True)
        out.flush()
        np.save(out_dir / "samples_tracks.npy", tracks)
        (out_dir / "run.json").write_text(json.dumps({**run, "model": str(mdir), "s0": s0, "samples": m}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/kaggle/ctx64_long",
         int(sys.argv[2]) if len(sys.argv) > 2 else 4,
         [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.3, 0.5, 0.7])
