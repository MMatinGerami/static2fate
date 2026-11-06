import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from static2fate.autoencoder import AutoEncoder  # noqa: E402
from static2fate.flow import UNet, flow_matching_loss, sample  # noqa: E402
from static2fate.latent import decode, encode, float_to_uint8, load_autoencoder, uint8_to_float  # noqa: E402
from static2fate.video import VideoUNet, sample_video, video_flow_loss  # noqa: E402


def test_unet_generates_four_channel_codes_with_four_channel_context():
    torch.manual_seed(0)
    m = UNet(ch=16, cdim=32, context=True, in_ch=4)
    z = torch.randn(2, 4, 32, 32)
    cond = {"ctx": torch.randn(2, 4, 32, 32), "ctx_frame": torch.tensor([0, 4])}
    assert torch.isfinite(flow_matching_loss(m, z, torch.tensor([6, 12]), torch.tensor([0, 1]), **cond))
    out = sample(m.eval(), torch.tensor([6, 12]), torch.tensor([0, 1]), steps=2, size=32, clamp=False, **cond)
    assert out.shape == (2, 4, 32, 32)
    assert out.abs().max() > 1  # latents are not clamped to the image range


def test_video_unet_in_latent_matches_frame_model_at_init():
    torch.manual_seed(0)
    m = VideoUNet(ch=16, cdim=32, in_ch=4).eval()
    x = torch.randn(2, 3, 4, 32, 32)
    ctx = torch.randn(2, 4, 32, 32)
    frame = torch.arange(6, 9)[None].expand(2, -1)
    fate, cf = torch.tensor([0, 1]), torch.tensor([0, 3])
    s = torch.tensor([0.3, 0.7])
    with torch.no_grad():
        v = m(x, s, frame, fate, ctx, cf)
        ref = torch.stack([UNet.forward(m, x[:, i], s, frame[:, i], fate, ctx=ctx, ctx_frame=cf) for i in range(3)], 1)
    assert v.shape == x.shape and torch.allclose(v, ref, atol=1e-5)
    assert torch.isfinite(video_flow_loss(m, x, frame, fate, ctx, cf))
    out = sample_video(m, frame, fate, ctx, cf, steps=2, clamp=False)
    assert out.shape == x.shape


def test_encode_decode_round_trip_shapes_and_scale(tmp_path):
    torch.manual_seed(0)
    ae = AutoEncoder(ch=16, z=4).eval()
    torch.save(ae.state_dict(), tmp_path / "model.pt")
    (tmp_path / "run.json").write_text(json.dumps({"ch": 16, "z": 4, "latent_std": 2.0, "factor": 4, "size": 64}))
    ae2, std, run = load_autoencoder(tmp_path, "cpu")
    assert std == 2.0 and run["factor"] == 4
    x8 = torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8)
    x = uint8_to_float(x8)
    assert x.min() >= -1 and x.max() <= 1 and torch.equal(float_to_uint8(x), x8)
    z = encode(ae2, x, std, batch=2)
    assert z.shape == (5, 4, 16, 16)
    with torch.no_grad():
        mean, _ = ae.encode(x)
    assert torch.allclose(z, mean / 2.0, atol=1e-5)  # divided by latent_std
    out = decode(ae2, z, std, batch=2)
    assert out.shape == x.shape and out.abs().max() <= 1


def test_latent_trainer_and_generator_agree_on_frames_and_layout(tmp_path):
    train_latent = importlib.import_module("34_train_latent")
    gen = importlib.import_module("35_generate_latent")
    assert train_latent.CLIP == list(range(6, 13)) and train_latent.MAX_CTX == 6
    assert all(f in train_latent.CLIP for f in gen.FUTURE) and gen.FUTURE == [6, 8, 10, 12]
    m = train_latent.build("video", 16, 4)
    assert isinstance(m, VideoUNet) and m.in_ch == 4
    m = train_latent.build("frame", 16, 4)
    assert isinstance(m, UNet) and m.context and m.in_ch == 4
    z = np.zeros((3, 13, 4, 8, 8), np.float16)
    np.savez(tmp_path / "train.npz", z=z, fate=np.array([0, 1, 0]), tracks=np.array(["a", "b", "c"]))
    zz, fate, tracks = train_latent.load_latents(tmp_path / "train.npz")
    assert zz.dtype == torch.float32 and zz.shape == (3, 13, 4, 8, 8) and fate.tolist() == [0, 1, 0]
    assert list(tracks) == ["a", "b", "c"]
