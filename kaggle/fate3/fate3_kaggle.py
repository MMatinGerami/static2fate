"""Kaggle job: the latent video model with three fate classes, 6 h training then sampling.

Takes the autoencoder of the static2fate-ae kernel ("ae128"), re-encodes the two-fate cells
(33_encode_latents.py, a few minutes), then trains from scratch, as the reference latent video
model was (the warm start from the frame model bought nothing, docs/results.md "Do more steps
or a warm start buy content?"), with the fate split into survives / dies AnnexinV only /
dies condensed (34_train_latent.py --fates 3): on GPU 0 "lat_video_fate3", seed 4. GPU 1
would otherwise idle for the same 6 h of quota, so it trains "lat_video_s4", the two-class
model with the same seed, batch and steps, the seed-matched control (the existing reference
"lat_video" is seed 3). Set RUNS=lat_video_fate3 in the environment to run only the first.
Each is followed by 8 futures per held-out cell decoded to 128 px (35_generate_latent.py,
which reads the class count from run.json). Outputs in /kaggle/working/<name>/. Push with:
kaggle kernels push -p kaggle/fate3 (after replacing USERNAME in kernel-metadata.json).
"""

import os
import subprocess
import threading
import time
from pathlib import Path

HOURS = float(os.environ.get("HOURS", 6))
AE = "ae128"
WORK = Path("/kaggle/working")
REPO = WORK / "static2fate"
INPUT = Path("/kaggle/input")

data = next(INPUT.rglob("crops_tracks.npy")).parent
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/MMatinGerami/static2fate", str(REPO)], check=True)
(WORK / "commit.txt").write_text(subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout)
found = [p for p in INPUT.rglob("model.pt") if p.parent.name == AE]
if len(found) != 1:
    raise FileNotFoundError(f"{AE}/model.pt: {found}")
ae_dir = found[0].parent
latents = WORK / "latents"
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0", STATIC2FATE_DATA=str(data),
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
with (WORK / "encode.log").open("a") as log:
    subprocess.run(["python", str(REPO / "scripts/33_encode_latents.py"), str(ae_dir), str(latents)],
                   env=env0, stdout=log, stderr=subprocess.STDOUT, check=True)
print((latents / "info.json").read_text(), flush=True)

COMMON = ["--ae", str(ae_dir), "--latents", str(latents), "--seed", "4", "--snapshot-every", "120",
          "--model", "video", "--batch", "16"]
RUNS = {"lat_video_fate3": [*COMMON, "--fates", "3"],
        "lat_video_s4": [*COMMON, "--fates", "2"]}
if os.environ.get("RUNS"):
    RUNS = {k: v for k, v in RUNS.items() if k in os.environ["RUNS"].split(",")}


def run(gpu, name, extra):
    env = dict(env0, CUDA_VISIBLE_DEVICES=str(gpu))
    with (WORK / f"{name}.log").open("a") as log:
        subprocess.run(["python", str(REPO / "scripts/34_train_latent.py"), "--device", "cuda", "--minutes",
                        str(HOURS * 60), "--out", str(WORK / name), *extra],
                       env=env, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(["python", str(REPO / "scripts/35_generate_latent.py"), str(WORK / name), "8"],
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
