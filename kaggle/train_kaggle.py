"""Kaggle job: train two context models at once, one per T4.

GPU 0: 128 px. GPU 1: 64 px, same setup as the one-hour Mac run but 8 h, to see what
longer training alone buys. Outputs (weights, logs, run.json, commit) end up in /kaggle/working.
Push with: kaggle kernels push -p kaggle   (after replacing USERNAME in kernel-metadata.json)
"""

import os
import subprocess
from pathlib import Path

HOURS = float(os.environ.get("HOURS", 8))  # Kaggle sessions stop at 9 h (12 h on some accounts)
WORK = Path("/kaggle/working")
REPO = WORK / "static2fate"

data = next(Path("/kaggle/input").rglob("crops_tracks.npy")).parent
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/MMatinGerami/static2fate", str(REPO)], check=True)
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
(WORK / "commit.txt").write_text(commit)
subprocess.run(["nvidia-smi"])

runs = {"ctx128": ["--size", "128", "--seed", "0"], "ctx64_long": ["--size", "64", "--seed", "1"]}
procs = []
for gpu, (name, extra) in enumerate(runs.items()):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), STATIC2FATE_DATA=str(data))
    cmd = ["python", str(REPO / "scripts/13_train_ctx.py"), "--device", "cuda", "--minutes", str(HOURS * 60),
           "--out", str(WORK / name), *extra]
    procs.append(subprocess.Popen(cmd, env=env, stdout=open(WORK / f"{name}.log", "w"), stderr=subprocess.STDOUT))
for p in procs:
    p.wait()
for name in runs:
    print(name, (WORK / f"{name}.log").read_text()[-400:])
