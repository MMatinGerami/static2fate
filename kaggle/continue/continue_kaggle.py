"""Kaggle job: continue the 64 px, 8 h context model for another 8 h, with and without the
perceptual term, one run per T4.

GPU 0: + VGG16 feature term (weight PERCEPTUAL). GPU 1: same continuation without it, the
control that separates "more training" from "the term". Both start from the ctx64_long
weights of the training job (kernel source). Outputs in /kaggle/working/<name>/.
Push with: kaggle kernels push -p kaggle/continue   (after replacing USERNAME in kernel-metadata.json)
"""

import os
import subprocess
import threading
import time
from pathlib import Path

HOURS = float(os.environ.get("HOURS", 8))
PERCEPTUAL = "0.5"  # the s-weighted term is about 0.17 against an MSE of 0.09, so 0.5 makes it count
WORK = Path("/kaggle/working")
REPO = WORK / "static2fate"

data = next(Path("/kaggle/input").rglob("crops_tracks.npy")).parent
init = next(Path("/kaggle/input").rglob("ctx64_long/model.pt"))
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/MMatinGerami/static2fate", str(REPO)], check=True)
(WORK / "commit.txt").write_text(subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout)
RUNS = {"ctx64_perc": ["--perceptual", PERCEPTUAL, "--seed", "2"], "ctx64_more": ["--seed", "2"]}


def run(gpu, name, extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), STATIC2FATE_DATA=str(data),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (WORK / f"{name}.log").open("w") as log:
        subprocess.run(["python", str(REPO / "scripts/13_train_ctx.py"), "--device", "cuda", "--minutes",
                        str(HOURS * 60), "--size", "64", "--init", str(init), "--out", str(WORK / name), *extra],
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
