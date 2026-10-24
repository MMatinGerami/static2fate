import numpy as np
import pandas as pd

from static2fate.divisions import condensation_events, find_divisions, recovery_after

FRAMES = list(range(1, 14))


def field(cells):
    """cells: {particle: (frames, xs, ys, areas, hoechst)}"""
    rows = []
    for pid, (fr, x, y, a, h) in cells.items():
        for i, f in enumerate(fr):
            rows.append({"frame": f, "particle": pid, "x": x[i], "y": y[i], "area": a[i], "hoechst_nuc_mean": h[i]})
    return pd.DataFrame(rows)


def test_condense_and_recover():
    a = np.array([1000, 1000, 1000, 400, 650, 640, 650.0])
    h = np.array([100, 100, 100, 200, 110, 110, 110.0])
    assert condensation_events(a, h) == [3]
    assert recovery_after(a, h, 3) == 4


def test_apoptosis_does_not_recover():
    a = np.array([1000, 1000, 1000, 500, 400, 300, 250.0])
    h = np.array([100, 100, 100, 180, 220, 250, 260.0])
    m = condensation_events(a, h)[0]
    assert recovery_after(a, h, m) is None


def test_division_needs_a_sister():
    n = len(FRAMES)
    area = [1000.0] * 5 + [400.0] + [650.0] * (n - 6)
    hoe = [100.0] * 5 + [200.0] + [110.0] * (n - 6)
    mother = (FRAMES, [100.0] * n, [100.0] * n, area, hoe)
    sister = (FRAMES[6:], [130.0] * (n - 6), [100.0] * (n - 6), [640.0] * (n - 6), [110.0] * (n - 6))
    far = (FRAMES[6:], [600.0] * (n - 6), [600.0] * (n - 6), [640.0] * (n - 6), [110.0] * (n - 6))
    with_sister = find_divisions(field({0: mother, 1: sister}), FRAMES).set_index("particle")
    assert with_sister.loc[0, "division"] and with_sister.loc[0, "sister"] == 1
    alone = find_divisions(field({0: mother, 2: far}), FRAMES).set_index("particle")
    assert alone.loc[0, "mitosis_like"] and not alone.loc[0, "division"]


def test_late_condensation_onset():
    n = len(FRAMES)
    area = [1000.0] * (n - 2) + [400.0, 380.0]
    hoe = [100.0] * (n - 2) + [200.0, 210.0]
    d = find_divisions(field({0: (FRAMES, [0.0] * n, [0.0] * n, area, hoe)}), FRAMES).iloc[0]
    assert d.condensed_at_end and d.end_onset_frame == 12 and not d.mitosis_like
