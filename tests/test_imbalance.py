import numpy as np
from faultdiag.imbalance import class_table, majority_baseline_accuracy, resample

rng = np.random.RandomState(0)
y = np.array([0] * 1700 + [1] * 100 + [2] * 120 + [3] * 80)
X = rng.randn(len(y), 5) + y[:, None]


def test_majority_baseline_and_table():
    assert round(majority_baseline_accuracy(y), 3) == 0.85
    t = class_table(y)
    assert t.rows.tolist() == [1700, 100, 120, 80] and t.imbalance_ratio.iloc[3] == 21.2


def test_undersample_shrinks_normal_only():
    _, yr, _ = resample("undersample", X, y, normal_cap=500)
    assert np.bincount(yr).tolist() == [500, 100, 120, 80]


def test_smote_grows_minorities_only():
    _, yr, _ = resample("smote", X, y)
    c = np.bincount(yr)
    assert c[0] == 1700 and min(c[1:]) >= 340


def test_class_weight_gives_rare_classes_more_weight():
    _, yr, w = resample("class_weight", X, y)
    assert w[yr == 3].mean() > w[yr == 0].mean()


def test_original_unchanged():
    Xr, yr, w = resample("original", X, y)
    assert len(yr) == len(y) and w is None
