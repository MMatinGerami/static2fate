"""Kaggle job: a twice-as-wide context model (ch 128, 35 M parameters), one run per T4, 8 h.

The 64-channel model has the same flow matching loss on held-out cells as on training cells
(scripts/27_fit_check.py) and did not improve with 8 more hours, so it is not memorising its
1,919 training cells: the next thing to test is capacity. GPU 0: ch 128 on the same cells.
GPU 1: ch 128 on all ten doses (5,860 training cells), to see whether more varied cells add
to the capacity. Held-out wells are the same in both, so 27_fit_check.py and the scoring chain
compare them directly with ctx64_long. EMA snapshots every 2 h follow held-out loss over
training. Outputs in /kaggle/working/<name>/.
Push with: kaggle kernels push -p kaggle/wide   (after replacing USERNAME in kernel-metadata.json)
"""

import os
import subprocess
import threading
import time
from pathlib import Path

HOURS = float(os.environ.get("HOURS", 8))
WORK = Path("/kaggle/working")
REPO = WORK / "static2fate"

data = next(Path("/kaggle/input").rglob("crops_tracks.npy")).parent
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/MMatinGerami/static2fate", str(REPO)], check=True)
(WORK / "commit.txt").write_text(subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout)
COMMON = ["--ch", "128", "--seed", "3", "--snapshot-every", "120"]
RUNS = {"ctx64_w128": COMMON, "ctx64_w128_all": [*COMMON, "--all-doses"]}


def run(gpu, name, extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), STATIC2FATE_DATA=str(data),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (WORK / f"{name}.log").open("w") as log:
        subprocess.run(["python", str(REPO / "scripts/13_train_ctx.py"), "--device", "cuda", "--minutes",
                        str(HOURS * 60), "--size", "64", "--out", str(WORK / name), *extra],
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
