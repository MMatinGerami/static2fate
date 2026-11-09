import importlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from static2fate.flow import UNet  # noqa: E402
from static2fate.video import VideoUNet, sample_video, video_flow_loss  # noqa: E402


def _inputs(b=2, t=4, size=32):
    x = torch.randn(b, t, 3, size, size)
    ctx = torch.randn(b, 3, size, size)
    frame = torch.arange(6, 6 + t)[None].expand(b, -1)
    return x, ctx, frame, torch.tensor([0, 1])[:b], torch.tensor([0, 3])[:b]


def test_video_model_equals_frame_model_at_init():
    torch.manual_seed(0)
    m = VideoUNet(ch=16, cdim=32).eval()
    x, ctx, frame, fate, cf = _inputs()
    s = torch.tensor([0.3, 0.7])
    with torch.no_grad():
        v = m(x, s, frame, fate, ctx, cf)
        ref = torch.stack([UNet.forward(m, x[:, i], s, frame[:, i], fate, ctx=ctx, ctx_frame=cf)
                           for i in range(x.shape[1])], 1)
    assert v.shape == x.shape
    assert torch.allclose(v, ref, atol=1e-5)  # temporal layers start at zero


def test_temporal_layers_mix_frames_after_training_step():
    torch.manual_seed(0)
    m = VideoUNet(ch=16, cdim=32)
    x, ctx, frame, fate, cf = _inputs()
    loss = video_flow_loss(m, x, frame, fate, ctx, cf)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.temporal.parameters())
    with torch.no_grad():
        for p in m.temporal.parameters():
            p.add_(0.1 * torch.randn_like(p))
    m.eval()
    with torch.no_grad():
        v = m(x, torch.tensor([0.5, 0.5]), frame, fate, ctx, cf)
        x2 = x.clone()
        x2[:, -1] += 1.0  # changing the last frame now changes the first frame's velocity
        v2 = m(x2, torch.tensor([0.5, 0.5]), frame, fate, ctx, cf)
    assert not torch.allclose(v[:, 0], v2[:, 0])


def test_sample_video_shape_and_range():
    torch.manual_seed(0)
    m = VideoUNet(ch=16, cdim=32).eval()
    x, ctx, frame, fate, cf = _inputs()
    out = sample_video(m, frame, fate, ctx, cf, steps=2)
    assert out.shape == x.shape and out.abs().max() <= 1


def test_clip_frames_and_context_are_ordered():
    train_video = importlib.import_module("28_train_video")
    assert train_video.CLIP == list(range(6, 13))
    gen = importlib.import_module("29_generate_clips")
    assert all(f in train_video.CLIP for f in gen.FUTURE)


def test_temporal_changes_on_smooth_and_jumping_clips():
    temporal = importlib.import_module("30_temporal")
    rng = np.random.default_rng(0)
    frames = temporal.CLIP
    base = rng.integers(0, 255, (3, 1, 1, 3, 64, 64), dtype=np.uint8).repeat(len(frames), 2)
    smooth = base.copy()
    jumping = base.copy()
    jumping[:, :, 1::2] = 255 - jumping[:, :, 1::2]  # every other frame inverted
    cs, cj = temporal.changes(smooth, frames, 1), temporal.changes(jumping, frames, 1)
    assert np.allclose(cs["pixel"], 0) and (cj["pixel"] > 1).all()
    c60 = temporal.changes(jumping, frames, 2)
    assert np.allclose(c60["pixel"], 0)  # frames two apart are identical here
    assert "area_reversals" in c60


def test_video_model_loads_frame_model_weights():
    torch.manual_seed(0)
    frame = UNet(ch=16, cdim=32, context=True)
    m = VideoUNet(ch=16, cdim=32)
    missing, unexpected = m.load_state_dict(frame.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("temporal.") for k in missing)
    x, ctx, fr, fate, cf = _inputs()
    s = torch.tensor([0.3, 0.7])
    m.eval()
    frame.eval()
    with torch.no_grad():
        v = m(x, s, fr, fate, ctx, cf)
        ref = torch.stack([frame(x[:, i], s, fr[:, i], fate, ctx=ctx, ctx_frame=cf) for i in range(x.shape[1])], 1)
    assert torch.allclose(v, ref, atol=1e-5)  # starts exactly as the frame model


def test_temporal_downsamples_128_px_clips_to_64():
    temporal = importlib.import_module("30_temporal")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, (2, 1, 3, 3, 128, 128), dtype=np.uint8)
    y = temporal.to64(x)
    assert y.shape == (2, 1, 3, 3, 64, 64) and y.dtype == np.uint8
    assert abs(float(y[0, 0, 0, 0, 0, 0]) - x[0, 0, 0, 0, :2, :2].mean()) <= 0.5
    assert temporal.to64(y) is not None and temporal.to64(y).shape[-1] == 64


def test_movie_helpers():
    movies = importlib.import_module("36_movies")
    assert movies.hhmm(0) == "0 h 00" and movies.hhmm(7) == "3 h 30" and movies.hhmm(12) == "6 h 00"
    small = np.zeros((3, 64, 64), np.uint8)
    assert movies.rgb(small).size == (movies.TILE, movies.TILE)
    big = np.zeros((3, 128, 128), np.uint8)
    big[0] = 255  # Hoechst goes to the blue channel
    assert movies.rgb(big).getpixel((0, 0)) == (0, 0, 255)


def test_field_constants_from_tiffs(tmp_path, monkeypatch):
    import tifffile

    scaling = importlib.import_module("37_field_scaling")
    rng = np.random.default_rng(0)
    for t in range(1, 14):
        tifffile.imwrite(tmp_path / f"W-01_F0001_T{t:04d}_Z0001_C01.tif", rng.integers(100, 1100, (64, 64), dtype=np.uint16))
    monkeypatch.setattr(scaling, "IMAGES", tmp_path)
    k = scaling.field_constants("W-01_F0001", {"hoechst": 1})
    assert set(k) == {"hoechst_lo", "hoechst_hi"} and 100 <= k["hoechst_lo"] < 150 and 1050 < k["hoechst_hi"] <= 1100
    assert scaling.field_constants("W-01_F0002", {"hoechst": 1}) == {}  # missing frames


def test_sample_video_project_hook_reaches_target():
    m = VideoUNet(ch=16, cdim=32).eval()
    for p in m.out.parameters():
        torch.nn.init.zeros_(p)
    x, ctx, frame, fate, cf = _inputs()
    z = torch.randn_like(x)
    out = sample_video(m, frame, fate, ctx, cf, steps=3, x0=z.clone(), clamp=False,
                       project=lambda y, s: (1 - s) * z + s * x)
    assert torch.allclose(out, x)

