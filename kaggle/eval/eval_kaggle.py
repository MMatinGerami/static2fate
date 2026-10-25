"""Kaggle job: energy-score evaluation of the two models from the training job, one per T4.

Attaches the training job's output (kernel source) and the crops dataset, runs
scripts/17_energy_score.py on each model with 8 samples per cell, and leaves the per-cell
scores and tables in /kaggle/working.
Push with: kaggle kernels push -p kaggle/eval   (after replacing USERNAME in kernel-metadata.json)
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "static2fate"

data = next(Path("/kaggle/input").rglob("crops_tracks.npy")).parent
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/MMatinGerami/static2fate", str(REPO)], check=True)
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
(WORK / "commit.txt").write_text(commit)

procs, names = [], ["ctx128", "ctx64_long"]
for gpu, name in enumerate(names):
    src = next(Path("/kaggle/input").rglob(f"{name}/model.pt")).parent
    shutil.copytree(src, WORK / name)  # the script writes its per-cell scores next to the model
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), STATIC2FATE_DATA=str(data))
    log = (WORK / f"eval_{name}.log").open("w")
    procs.append(subprocess.Popen(["python", str(REPO / "scripts/17_energy_score.py"), str(WORK / name), "8"],
                                  env=env, stdout=log, stderr=subprocess.STDOUT))

wait = 180
while any(p.poll() is None for p in procs):
    time.sleep(wait)
    wait = 600
    for name, p in zip(names, procs):
        lines = (WORK / f"eval_{name}.log").read_text().strip().splitlines()
        print(name, "running" if p.poll() is None else f"exited {p.returncode}", lines[-1] if lines else "", flush=True)
for f in (REPO / "results/tables").glob("energy_score_*.csv"):
    shutil.copy(f, WORK / f.name)
