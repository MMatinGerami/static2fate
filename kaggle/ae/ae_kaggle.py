"""Kaggle job: the autoencoder (31_train_autoencoder.py), one run per T4, 5 h training.

GPU 0 ("ae128"): L1 + perceptual + KL on every frame of every training cell at all ten doses
(5,860 cells, 76k images). GPU 1 ("ae128_gan"): the same plus a PatchGAN hinge term (weight
0.1) from step 5,000 on. After training each GPU reconstructs the held-out cells
(32_reconstruct.py), which writes the reconstruction in the layout the scoring chain reads,
so the probe can be run on it (the ceiling for generation in that latent). Outputs in
/kaggle/working/<name>/. Push with: kaggle kernels push -p kaggle/ae
(after replacing USERNAME in kernel-metadata.json). 5 h plus reconstruction stays well
under the 12 h session limit.
"""

import os
import subprocess
import threading
import time
from pathlib import Path

HOURS = float(os.environ.get("HOURS", 5))
WORK = Path("/kaggle/working")
REPO = WORK / "static2fate"

data = next(Path("/kaggle/input").rglob("crops_tracks.npy")).parent
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/MMatinGerami/static2fate", str(REPO)], check=True)
(WORK / "commit.txt").write_text(subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout)
COMMON = ["--seed", "3", "--snapshot-every", "120", "--batch", "32"]
RUNS = {"ae128": COMMON, "ae128_gan": [*COMMON, "--gan", "0.1", "--gan-start", "5000"]}


def run(gpu, name, extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), STATIC2FATE_DATA=str(data),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (WORK / f"{name}.log").open("a") as log:
        subprocess.run(["python", str(REPO / "scripts/31_train_autoencoder.py"), "--device", "cuda", "--minutes",
                        str(HOURS * 60), "--out", str(WORK / name), *extra],
                       env=env, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(["python", str(REPO / "scripts/32_reconstruct.py"), str(WORK / name), name],
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
