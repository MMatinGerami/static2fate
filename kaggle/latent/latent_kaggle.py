"""Kaggle job: the flow models in the autoencoder's latent, one per T4, 6 h training then sampling.

Takes the autoencoder of the static2fate-ae kernel (AE, default "ae128"; "ae128_gan" for the
adversarial one), encodes every frame of the two-fate cells (33_encode_latents.py, a few
minutes), then trains on GPU 0 the frame model ("lat_frame", 34_train_latent.py --model
frame, batch 64) and on GPU 1 the video model from scratch ("lat_video", --model video,
batch 16). Each is followed by 8 futures per held-out cell decoded to 128 px
(35_generate_latent.py), in the layout the scoring chain reads. Outputs in
/kaggle/working/<name>/. Push with: kaggle kernels push -p kaggle/latent (after replacing
USERNAME in kernel-metadata.json). Sampling in the 32 px latent is much cheaper than at
64 px, so 6 h of training plus sampling stays well under the 12 h session limit.
"""

import os
import subprocess
import threading
import time
from pathlib import Path

HOURS = float(os.environ.get("HOURS", 6))
AE = os.environ.get("AE", "ae128")
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

COMMON = ["--ae", str(ae_dir), "--latents", str(latents), "--seed", "3", "--snapshot-every", "120"]
RUNS = {"lat_frame": [*COMMON, "--model", "frame", "--batch", "64"],
        "lat_video": [*COMMON, "--model", "video", "--batch", "16"]}


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
