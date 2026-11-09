import numpy as np

from birdclef.utils import macro_auc


def test_macro_auc_skips_classes_without_positives_or_negatives():
    y = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1]])
    s = np.array([[0.9, 0.1, 0.2], [0.2, 0.5, 0.3], [0.8, 0.2, 0.9], [0.1, 0.4, 0.1]])
    mean, per_class = macro_auc(y, s)
    assert per_class[0] == 1.0
    assert np.isnan(per_class[1]) and np.isnan(per_class[2])  # no positives / no negatives
    assert mean == 1.0
    assert macro_auc(y, -s)[0] == 0.0
