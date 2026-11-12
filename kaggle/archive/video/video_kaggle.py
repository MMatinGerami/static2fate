"""Kaggle job: the video model (28_train_video.py), one run per T4, 7 h training then sampling.

GPU 0: ch 64 on the two-fate doses (1,919 training cells), the direct counterpart of
ctx64_long, which draws the same frames one at a time. GPU 1: ch 64 on all ten doses (5,860
cells). After training each GPU generates 8 clips per held-out cell (29_generate_clips.py),
written in the layout the scoring chain reads (20, 22, 26, 30). Outputs in
/kaggle/working/<name>/. Push with: kaggle kernels push -p kaggle/video
(after replacing USERNAME in kernel-metadata.json).
"""

import os
import subprocess
import threading
import time
from pathlib import Path

HOURS = float(os.environ.get("HOURS", 7))
WORK = Path("/kaggle/working")
REPO = WORK / "static2fate"

data = next(Path("/kaggle/input").rglob("crops_tracks.npy")).parent
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/MMatinGerami/static2fate", str(REPO)], check=True)
(WORK / "commit.txt").write_text(subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout)
COMMON = ["--seed", "3", "--snapshot-every", "120"]
RUNS = {"video64": COMMON, "video64_all": [*COMMON, "--all-doses"]}


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
