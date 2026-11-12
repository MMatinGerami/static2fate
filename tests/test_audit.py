"""Label audit helpers (static2fate.audit) and the display convention (static2fate.display)."""

import numpy as np

from static2fate.audit import background_share, cumulative_incidence, sustained_onset
from static2fate.display import composite, mosaic, unit, window


def test_background_share_is_expected_background_over_label_count():
    assert background_share(50, 500, 0.1) == 1.0
    assert abs(background_share(200, 500, 0.1) - 0.25) < 1e-12
    assert np.isnan(background_share(0, 500, 0.1))


def test_sustained_onset_ignores_a_transient_crossing():
    mask = np.array([[0, 1, 0, 1, 1],   # rounds up once, recovers, then stays: onset at index 3
                     [0, 0, 0, 0, 0],   # never
                     [0, 1, 1, 0, 0],   # transient only: no onset
                     [1, 1, 1, 1, 1]], bool)
    onset = sustained_onset(mask)
    assert onset[0] == 3 and onset[3] == 0
    assert np.isnan(onset[1]) and np.isnan(onset[2])


def test_cumulative_incidence_counts_cells_past_onset():
    ci = cumulative_incidence(np.array([2.0, 4.0, np.nan, np.nan]), 5)
    assert ci.tolist() == [0.0, 0.25, 0.25, 0.5, 0.5]


def test_unit_and_composite_put_561_in_red_and_hoechst_in_blue():
    assert unit(np.array([0, 50, 100, 200]), 0, 100).tolist() == [0.0, 0.5, 1.0, 1.0]
    one, zero = np.full((2, 2), 100), np.zeros((2, 2))
    rng = {"hoechst": (0, 100), "cl488a": (0, 100), "cl561": (0, 100)}
    rgb = composite(hoechst=one, green=zero, red=zero, ranges=rng)
    assert rgb[0, 0].tolist() == [0, 0, 255]
    rgb = composite(hoechst=zero, green=zero, red=one, ranges=rng)
    assert rgb[0, 0].tolist() == [255, 0, 0]


def test_window_pads_at_the_border_and_mosaic_places_tiles():
    img = np.arange(16).reshape(4, 4)
    w = window(img, 0, 0, 2)
    assert w.shape == (2, 2) and w[1, 1] == img[0, 0] and w[0, 0] == 0
    tile = np.full((2, 2, 3), 9, np.uint8)
    out, tops, lefts = mosaic([[tile, None], [tile, tile]], gap=1)
    assert out.shape == (5, 5, 3) and tops.tolist() == [0, 3] and lefts.tolist() == [0, 3]
    assert out[0, 3].tolist() == [0, 0, 0] and out[3, 3].tolist() == [9, 9, 9]
