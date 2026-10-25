import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from static2fate.evaluate import field_bootstrap, group_of  # noqa: E402
from static2fate.flow import UNet, flow_matching_loss, invert, sample  # noqa: E402
from static2fate.readouts import cell_readouts  # noqa: E402


def test_context_model_shapes_and_sizes():
    torch.manual_seed(0)
    m = UNet(ch=16, cdim=32, context=True)
    for size in (64, 128):
        x = torch.randn(2, 3, size, size)
        cond = {"ctx": x, "ctx_frame": torch.tensor([0, 4])}
        loss = flow_matching_loss(m, x, torch.tensor([6, 12]), torch.tensor([0, 1]), **cond)
        assert torch.isfinite(loss)
        out = sample(m, torch.tensor([6, 12]), torch.tensor([0, 1]), steps=2, size=size, **cond)
        assert out.shape == (2, 3, size, size) and out.abs().max() <= 1


def test_sampler_is_guided_mix_of_two_passes():
    torch.manual_seed(0)
    m = UNet(ch=16, cdim=32).eval()
    frame, fate = torch.tensor([3, 9]), torch.tensor([0, 1])
    z = torch.randn(2, 3, 64, 64)
    x = z.clone()
    with torch.no_grad():
        for i in range(3):
            s = torch.full((2,), i / 3)
            vc, vu = m(x, s, frame, fate), m(x, s, frame, torch.full_like(fate, 2))
            x = x + (vu + 2.0 * (vc - vu)) / 3
    assert torch.allclose(sample(m, frame, fate, steps=3, x0=z.clone()), x.clamp(-1, 1), atol=1e-4)


def test_invert_then_sample_is_identity_for_zero_velocity():
    m = UNet(ch=16, cdim=32).eval()
    for p in m.out.parameters():
        torch.nn.init.zeros_(p)  # velocity 0 everywhere
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    z = invert(m, x, torch.zeros(2, dtype=torch.long), steps=4)
    assert torch.allclose(z, x)


def test_context_pairs_are_ordered():
    train_ctx = importlib.import_module("13_train_ctx")
    torch.manual_seed(0)
    _, tc, t = train_ctx.pairs(10, 5000)
    assert tc.min() == 0 and tc.max() == train_ctx.MAX_CTX
    assert (t > tc).all() and t.max() == 12


def test_readouts_find_central_nucleus():
    im = -np.ones((1, 3, 64, 64))
    yy, xx = np.mgrid[:64, :64]
    disk = np.hypot(yy - 32, xx - 32) < 8
    im[0, 0][disk] = 0.8
    im[0, 0][np.hypot(yy - 5, xx - 5) < 3] = 0.8  # neighbour at the corner
    im[0, 1][~disk] = 0.2
    r = cell_readouts(im)[0]
    assert abs(r[0] - disk.sum()) < 1 and abs(r[1] - 0.8) < 1e-6 and abs(r[2] - 0.2) < 1e-6


def test_field_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"field": np.repeat([f"f{i}" for i in range(12)], 20), "g": "a",
                      "y": rng.random(240) < 0.7})
    d["y"] = d.y.astype(float)
    t = field_bootstrap(d, "y", ["g"], n_boot=300).iloc[0]
    assert t.ci_low <= t.estimate <= t.ci_high and t.n == 240
    assert group_of(0, True) == "survives" and group_of(1, False) == "dies, AnnexinV only"


def test_energy_score():
    from static2fate.evaluate import energy_score

    y = np.zeros(2)
    assert energy_score(np.array([[3.0, 4.0]]), y) == 5.0  # one sample: plain distance
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(4000, 2))
    calibrated = np.mean([energy_score(rng.normal(size=(16, 2)), t) for t in truth[:500]])
    too_narrow = np.mean([energy_score(0.2 * rng.normal(size=(16, 2)), t) for t in truth[:500]])
    point = np.mean([energy_score(np.zeros((1, 2)), t) for t in truth[:500]])
    assert calibrated < too_narrow and calibrated < point
    assert np.isnan(energy_score(np.full((3, 2), np.nan), y))


def test_rank_of_own():
    from static2fate.evaluate import rank_of_own

    cand = np.array([[0.0, 0], [10, 0], [20, 0], [30, 0]])
    q = np.array([[[0.1, 0], [12, 0]],  # cell 0: own nearest, then behind [10,0] and [20,0]
                  [[29, 0], [29, 0]]])  # cell 3 (own row 3): nearest both times
    pct, top1 = rank_of_own(q, cand, np.array([0, 3]))
    assert top1.tolist() == [[True, False], [True, True]]
    assert pct[0].tolist() == [0.0, 2 / 3] and pct[1].tolist() == [0.0, 0.0]
    assert pct.shape == (2, 2)
