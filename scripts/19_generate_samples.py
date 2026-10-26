"""Generate and save M futures per held-out cell from a context model (0 h frame, true fate).

Same draws as 17_energy_score.py but the images are kept, as uint8, so new comparisons
(20_feature_score.py) do not need the GPU again. For a paired-flow model (run.json has
"sigma", 25_train_paired.py) the integration starts from the 0 h image plus sigma x noise.

Usage: uv run scripts/19_generate_samples.py [model dir] [M, default 8] [cells: n or start:stop]
Writes <model dir>/samples[_start_stop].npy, shape (cells, 4 frames, M, 3, s, s), and the
matching samples[_start_stop]_tracks.npy. Frames are 6, 8, 10, 12 (3 to 6 h).
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
CELLS_PER_BATCH = {64: 32, 128: 8}


def main(mdir: Path, m: int, cells: str | None) -> None:
    train = importlib.import_module("09_train_flow")
    run = json.loads((mdir / "run.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(ch=run["ch"], context=True).to(device)
    model.load_state_dict(torch.load(mdir / "model.pt", map_location=device, weights_only=True))
    model.eval()
    x, fate, tracks = train.load("test", run["size"])
    suffix = ""
    if cells:
        a, _, b = cells.partition(":")
        sel = slice(int(a), int(b)) if b else slice(0, int(a))
        x, fate, tracks = x[sel], fate[sel], tracks[sel]
        suffix = f"_{sel.start}_{sel.stop}"
    n, size = len(x), run["size"]
    print(f"cells {n}, {m} samples each, size {size}, device {device}", flush=True)
    amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()
    sigma = run.get("sigma")  # paired flow: start from the context image plus noise

    out = np.lib.format.open_memmap(mdir / f"samples{suffix}.npy", mode="w+", dtype=np.uint8,
                                    shape=(n, len(FUTURE), m, 3, size, size))
    g = torch.Generator().manual_seed(0)
    per_batch = CELLS_PER_BATCH[size]
    for b in range(0, n, per_batch):
        sl = slice(b, min(b + per_batch, n))
        k = sl.stop - sl.start
        ctx = x[sl, 0].repeat_interleave(m, 0).to(device)
        fk = fate[sl].repeat_interleave(m).to(device)
        for j, f in enumerate(FUTURE):
            z = torch.randn(k * m, 3, size, size, generator=g).to(device)
            if sigma is not None:
                z = ctx + sigma * z
            with amp:
                im = sample(model, torch.full((k * m,), f, device=device), fk, x0=z, ctx=ctx,
                            ctx_frame=torch.zeros(k * m, dtype=torch.long, device=device)).float().cpu()
            out[sl, j] = ((im + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).view(k, m, 3, size, size).numpy()
        print(f"cells {sl.stop}/{n}", flush=True)
    out.flush()
    np.save(mdir / f"samples{suffix}_tracks.npy", tracks)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/flow_ctx",
         int(sys.argv[2]) if len(sys.argv) > 2 else 8,
         sys.argv[3] if len(sys.argv) > 3 else None)
