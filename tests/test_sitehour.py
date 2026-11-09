import numpy as np

from birdclef.sitehour import apply, priors, site_hour


def test_site_hour_from_name():
    assert site_hour("BC2026_Train_0001_S08_20250606_030007.ogg") == ("S08", 3)
    assert site_hour("H02_20230420_174000") == ("H02", 17)


def test_priors_multiply_by_site_and_hour_means():
    names = ["S1_20250101_010000", "S1_20250101_020000", "S2_20250101_010000"]
    probs = np.array([[[0.2, 0.4]], [[0.6, 0.8]], [[1.0, 0.0]]])  # [recordings, windows, classes]
    sites, hours = map(np.array, zip(*map(site_hour, names)))
    site, hour = priors(probs, sites), priors(probs, hours)
    assert np.allclose(site["S1"], [0.4, 0.6]) and np.allclose(hour[1], [0.6, 0.2])
    out = apply(probs, names, site, hour)
    assert np.allclose(out[0, 0], [0.2 * 0.4 * 0.6, 0.4 * 0.6 * 0.2])
    unseen = apply(probs[:1], ["S9_20250101_230000"], site, hour)  # unseen site and hour: the overall mean
    assert np.allclose(unseen[0, 0], probs[0, 0] * probs.mean((0, 1)) ** 2)
