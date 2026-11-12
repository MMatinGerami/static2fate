"""Kaggle job: second video round, one run per T4, 8 h training then sampling.

Is the video model's per-frame gap to the frame model only training budget? GPU 0
("video64_init"): the video model initialised from the frame model ctx64_long (its temporal
layers start at zero, so it begins exactly as that model and only has to learn the coupling).
GPU 1 ("video64_cont"): the 7 h video model video64 continued, 16 h in all. Both on the
two-fate doses, then 8 clips per held-out cell (29_generate_clips.py). Outputs in
/kaggle/working/<name>/. Push with: kaggle kernels push -p kaggle/video2
(after replacing USERNAME in kernel-metadata.json).

Sampling 816 cells at 8 clips takes about 3.4 h on a T4 and a session ends at 12 h, so
training must stay at 8 h or less: the first run (9 h) was cut during sampling.
"""

import os
import subprocess
import threading
import time
from pathlib import Path

HOURS = float(os.environ.get("HOURS", 8))
WORK = Path("/kaggle/working")
REPO = WORK / "static2fate"
INPUT = Path("/kaggle/input")

data = next(INPUT.rglob("crops_tracks.npy")).parent
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/MMatinGerami/static2fate", str(REPO)], check=True)
(WORK / "commit.txt").write_text(subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout)


def checkpoint(name):
    found = [p for p in INPUT.rglob("model.pt") if p.parent.name == name]
    if len(found) != 1:
        raise FileNotFoundError(f"{name}/model.pt: {found}")
    return str(found[0])


COMMON = ["--seed", "3", "--snapshot-every", "120"]
RUNS = {"video64_init": [*COMMON, "--init", checkpoint("ctx64_long")],
        "video64_cont": [*COMMON, "--init", checkpoint("video64")]}
print({k: v[-1] for k, v in RUNS.items()}, flush=True)


def run(gpu, name, extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), STATIC2FATE_DATA=str(data),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (WORK / f"{name}.log").open("a") as log:
        subprocess.run(["python", str(REPO / "scripts/archive/28_train_video.py"), "--device", "cuda", "--minutes",
                        str(HOURS * 60), "--size", "64", "--out", str(WORK / name), *extra],
                       env=env, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(["python", str(REPO / "scripts/archive/29_generate_clips.py"), str(WORK / name), "8"],
                       env=env, stdout=log, stderr=subprocess.STDOUT)


threads = [threading.Thread(target=run, args=(gpu, name, extra)) for gpu, (name, extra) in enumerate(RUNS.items())]
for t in threads:
    t.start()
wait = 180
while any(t.is_alive() for t in threads):
    time.sleep(wait)
    wait = 600
    for name in RUNS:
        lines = (WORK / f"{name}.log").read_text().strip().splitlines()
        print(name, lines[-1] if lines else "", flush=True)
