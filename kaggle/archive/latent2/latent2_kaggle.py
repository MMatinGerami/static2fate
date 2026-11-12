"""Kaggle job: the second latent round, one run per T4, 6 h training then sampling.

Takes the autoencoder of the static2fate-ae kernel ("ae128") and the latent frame model of
the static2fate-latent kernel ("lat_frame"), re-encodes the two-fate cells
(33_encode_latents.py, a few minutes), then trains on GPU 0 the latent video model started
from the latent frame model ("lat_video_init", 34_train_latent.py --model video --init, the
recipe that worked in pixel space) and on GPU 1 the latent frame model continued for 6 h
more ("lat_frame_cont", --model frame --init), a direct test of whether more steps buy
content. Each is followed by 8 futures per held-out cell decoded to 128 px
(35_generate_latent.py). Outputs in /kaggle/working/<name>/. Push with:
kaggle kernels push -p kaggle/latent2 (after replacing USERNAME in kernel-metadata.json).
"""

import os
import subprocess
import threading
import time
from pathlib import Path

HOURS = float(os.environ.get("HOURS", 6))
AE = "ae128"
FRAME = "lat_frame"
WORK = Path("/kaggle/working")
REPO = WORK / "static2fate"
INPUT = Path("/kaggle/input")

data = next(INPUT.rglob("crops_tracks.npy")).parent
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/MMatinGerami/static2fate", str(REPO)], check=True)
(WORK / "commit.txt").write_text(subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout)


def one(name: str) -> Path:
    found = [p for p in INPUT.rglob("model.pt") if p.parent.name == name]
    if len(found) != 1:
        raise FileNotFoundError(f"{name}/model.pt: {found}")
    return found[0]


ae_dir = one(AE).parent
frame_model = one(FRAME)
latents = WORK / "latents"
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0", STATIC2FATE_DATA=str(data),
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
with (WORK / "encode.log").open("a") as log:
    subprocess.run(["python", str(REPO / "scripts/33_encode_latents.py"), str(ae_dir), str(latents)],
                   env=env0, stdout=log, stderr=subprocess.STDOUT, check=True)
print((latents / "info.json").read_text(), flush=True)

COMMON = ["--ae", str(ae_dir), "--latents", str(latents), "--seed", "4", "--snapshot-every", "120",
          "--init", str(frame_model)]
RUNS = {"lat_video_init": [*COMMON, "--model", "video", "--batch", "16"],
        "lat_frame_cont": [*COMMON, "--model", "frame", "--batch", "64"]}


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
