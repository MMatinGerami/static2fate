"""Generate and save M clips per held-out cell from a video model (0 h frame, true fate).

Each clip is the seven frames 6 to 12 (3 to 6 h) drawn jointly. Two files are written so the
existing scoring chain runs unchanged: clips.npy keeps every frame, shape
(cells, M, 7, 3, s, s) uint8, for the temporal comparisons of 30_temporal.py; samples.npy keeps
frames 6, 8, 10, 12 in the layout of 19_generate_samples.py, (cells, 4, M, 3, s, s), for
20_feature_score.py, 22_identity.py and 26_percell_skill.py. Same seed and cell order as 19.

Usage: uv run scripts/archive/29_generate_clips.py [model dir] [M, default 8] [cells: n or start:stop]

Cells are drawn 4 at a time on CUDA and 1 at a time on the Mac, where the Apple GPU allocator
is also capped at half the recommended working set and emptied after each batch (4 cells of
8 clips, 448 images per guided step, grew to 16 GB of unified memory and swapped). The noise
stream restarts at seed 0 for every invocation, so a part file's noise depends on its start
cell and batch size, not on the other parts.
"""

import contextlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))  # scripts/, for the current pipeline
from static2fate.video import VideoUNet, sample_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FUTURE = [6, 8, 10, 12]
CELLS_PER_BATCH = 4


def main(mdir: Path, m: int, cells: str | None) -> None:
    train = importlib.import_module("09_train_flow")
    run = json.loads((mdir / "run.json").read_text())
    clip = run["clip"]
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    per_batch = CELLS_PER_BATCH if device == "cuda" else 1
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.5)
    model = VideoUNet(ch=run["ch"]).to(device)
    model.load_state_dict(torch.load(mdir / "model.pt", map_location=device, weights_only=True))
    model.eval()
    x, fate, tracks = train.load("test", run["size"], frames=[0])
    suffix = ""
    if cells:
        a, _, b = cells.partition(":")
        sel = slice(int(a), int(b)) if b else slice(0, int(a))
        x, fate, tracks = x[sel], fate[sel], tracks[sel]
        suffix = f"_{sel.start}_{sel.stop}"
    n, size, t = len(x), run["size"], len(clip)
    print(f"cells {n}, {m} clips each, {t} frames, size {size}, device {device}", flush=True)
    amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()
    keep = [clip.index(f) for f in FUTURE]

    clips = np.lib.format.open_memmap(mdir / f"clips{suffix}.npy", mode="w+", dtype=np.uint8,
                                      shape=(n, m, t, 3, size, size))
    samples = np.lib.format.open_memmap(mdir / f"samples{suffix}.npy", mode="w+", dtype=np.uint8,
                                        shape=(n, len(FUTURE), m, 3, size, size))
    g = torch.Generator().manual_seed(0)
    for b in range(0, n, per_batch):
        sl = slice(b, min(b + per_batch, n))
        k = sl.stop - sl.start
        ctx = x[sl, 0].repeat_interleave(m, 0).to(device)
        fk = fate[sl].repeat_interleave(m).to(device)
        frame = torch.tensor(clip, device=device)[None].expand(k * m, -1)
        z = torch.randn(k * m, t, 3, size, size, generator=g).to(device)
        with amp:
            out = sample_video(model, frame, fk, ctx, torch.zeros(k * m, dtype=torch.long, device=device), x0=z)
        u8 = ((out.float().cpu() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).view(k, m, t, 3, size, size)
        clips[sl] = u8.numpy()
        samples[sl] = u8[:, :, keep].transpose(1, 2).numpy()
        if device == "mps":
            torch.mps.empty_cache()
        if sl.stop % 4 == 0 or sl.stop == n:
            print(f"cells {sl.stop}/{n}", flush=True)
    clips.flush()
    samples.flush()
    np.save(mdir / f"samples{suffix}_tracks.npy", tracks)
    np.save(mdir / f"clips{suffix}_tracks.npy", tracks)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/flow_video",
         int(sys.argv[2]) if len(sys.argv) > 2 else 8,
         sys.argv[3] if len(sys.argv) > 3 else None)
