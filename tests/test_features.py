import numpy as np

from static2fate.features import FEATURES_HAND, crop_features


def disk_crop(s: int, r: float, nuc_val: float = 0.8, ring_val: float = 0.2):
    im = -np.ones((3, s, s))
    yy, xx = np.mgrid[:s, :s]
    d = np.hypot(yy - s / 2 + 0.5, xx - s / 2 + 0.5)
    im[0][d < r] = nuc_val
    im[1][(d >= r) & (d < r + 20)] = ring_val  # green cytoplasm around the nucleus
    return im, (d < r).sum()


def test_crop_features_disk_at_two_sizes():
    for s, r in ((64, 8), (128, 16)):
        im, area = disk_crop(s, r)
        f = crop_features(im[None])[0]
        assert f.shape == (len(FEATURES_HAND),)
        assert abs(f[0] - area) < 2  # area
        assert f[1] < 0.3 and f[2] > 0.95  # round and solid
        assert abs(f[3] - 0.8) < 1e-6 and abs(f[4] + 1) < 1e-6  # Hoechst in nucleus, none in ring
        assert abs(f[6] - 0.2) < 1e-6  # green ring


def test_crop_features_empty_image_is_nan():
    im = np.full((1, 3, 64, 64), -1.0)
    im[0, 0, 0, 0] = 1.0  # a single bright pixel so Otsu has two classes; object far from centre
    f = crop_features(im)
    assert f.shape == (1, 9)
    assert np.isfinite(f[0, 0])  # the lone pixel is still the nearest object
    assert np.isnan(crop_features(np.zeros((1, 3, 64, 64)))).all()  # constant image: no object
