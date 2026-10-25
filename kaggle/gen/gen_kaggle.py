"""Kaggle job: save 8 futures per held-out cell from the two models of the training job.

One model per T4: 64 px for 98,000 steps on GPU 0, 128 px on GPU 1 (about 2 and 4 h). Uses
scripts/19_generate_samples.py; the sample arrays end up in /kaggle/working/<model>/ and are
scored on the Mac by scripts/20_feature_score.py.
Push with: kaggle kernels push -p kaggle/gen   (after replacing USERNAME in kernel-metadata.json)
"""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "static2fate"
MODELS = ["ctx64_long", "ctx128"]

data = next(Path("/kaggle/input").rglob("crops_tracks.npy")).parent
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/MMatinGerami/static2fate", str(REPO)], check=True)
(WORK / "commit.txt").write_text(subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout)


def generate(gpu, name):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), STATIC2FATE_DATA=str(data),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    src = next(Path("/kaggle/input").rglob(f"{name}/model.pt")).parent
    shutil.copytree(src, WORK / name)
    with (WORK / f"gen_{name}.log").open("w") as log:
        subprocess.run(["python", str(REPO / "scripts/19_generate_samples.py"), str(WORK / name), "8"],
                       env=env, stdout=log, stderr=subprocess.STDOUT)


threads = [threading.Thread(target=generate, args=(gpu, name)) for gpu, name in enumerate(MODELS)]
for t in threads:
    t.start()
wait = 180
while any(t.is_alive() for t in threads):
    time.sleep(wait)
    wait = 600
    for log in sorted(WORK.glob("gen_*.log")):
        lines = log.read_text().strip().splitlines()
        print(log.stem, lines[-1] if lines else "", flush=True)
for name in MODELS:  # the weights are already in the training job's output
    (WORK / name / "model.pt").unlink(missing_ok=True)
