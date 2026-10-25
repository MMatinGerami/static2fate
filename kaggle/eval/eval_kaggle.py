"""Kaggle job: energy-score evaluation of the two models from the training job, one per T4.

Attaches the training job's output (kernel source) and the crops dataset, runs
scripts/17_energy_score.py on each model with 8 samples per cell, and leaves the per-cell
scores and tables in /kaggle/working. The 128 px model takes about 8 h for all cells on one
T4, so its cells are split over the two GPUs (readouts are standardised on all cells in both
halves); the halves are joined by scripts/18_compare_models.py.
Push with: kaggle kernels push -p kaggle/eval   (after replacing USERNAME in kernel-metadata.json)
"""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "static2fate"

data = next(Path("/kaggle/input").rglob("crops_tracks.npy")).parent
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/MMatinGerami/static2fate", str(REPO)], check=True)
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
(WORK / "commit.txt").write_text(commit)

# one queue per GPU; the 128 px model is split in two halves of the 816 held-out cells
QUEUES = [[("ctx128_a", "ctx128", "0:408")],
          [("ctx128_b", "ctx128", "408:816"), ("ctx64_long", "ctx64_long", "0:816")]]


def run_queue(gpu, queue):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), STATIC2FATE_DATA=str(data),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for out, model, cells in queue:
        src = next(Path("/kaggle/input").rglob(f"{model}/model.pt")).parent
        shutil.copytree(src, WORK / out)  # the script writes its per-cell scores next to the model
        with (WORK / f"eval_{out}.log").open("w") as log:
            subprocess.run(["python", str(REPO / "scripts/17_energy_score.py"), str(WORK / out), "8", cells],
                           env=env, stdout=log, stderr=subprocess.STDOUT)


threads = [threading.Thread(target=run_queue, args=(gpu, q)) for gpu, q in enumerate(QUEUES)]
for t in threads:
    t.start()
wait = 180
while any(t.is_alive() for t in threads):
    time.sleep(wait)
    wait = 600
    for log in sorted(WORK.glob("eval_*.log")):
        lines = log.read_text().strip().splitlines()
        print(log.stem, lines[-1] if lines else "", flush=True)
for f in (REPO / "results/tables").glob("energy_score_*.csv"):
    shutil.copy(f, WORK / f.name)
