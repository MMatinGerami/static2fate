"""Green over red ratio and its log2 change (44_green_red.py)."""

import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/archive"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

gr = importlib.import_module("44_green_red")


def test_ratio_uses_eps_and_marks_bad_values():
    r = gr.ratio([2.0, 4.0, -1.0, 1.0], [1.0, 0.0, 1.0, np.nan], eps=1.0)
    assert r[0] == 1.0 and r[1] == 4.0
    assert np.isnan(r[2]) and np.isnan(r[3])


def test_log2_change_is_zero_for_no_change_and_nan_for_nonpositive():
    d = gr.log2_change([1.0, 4.0, 0.5, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 0.0])
    assert d[0] == 0 and d[1] == 2 and d[2] == -1
    assert np.isnan(d[3]) and np.isnan(d[4])


def test_summarise_counts_rises_past_threshold():
    import pandas as pd

    d = pd.DataFrame({"group": ["a"] * 4, "field": ["f1", "f1", "f2", "f2"], "dlog": [1.0, 0.0, 2.0, np.nan]})
    s = gr.summarise(d, ["group"])
    rises = s[s["measure"] == "rises"].iloc[0]
    assert rises["n"] == 3 and abs(rises["estimate"] - 2 / 3) < 1e-9
    assert s[s["measure"] == "dlog"].iloc[0]["median_dlog"] == 1.0
