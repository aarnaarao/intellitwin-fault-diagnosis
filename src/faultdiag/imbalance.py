"""Class-imbalance analysis and the five handling strategies. Apply to TRAINING data only."""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_sample_weight

from faultdiag.data import CLASSES

STRATEGIES = ["original", "class_weight", "undersample", "smote", "hybrid"]


def class_table(y):
    """Counts, share and imbalance ratio (majority count / class count) per class."""
    counts = np.bincount(y, minlength=len(CLASSES))
    return pd.DataFrame({"rows": counts, "share_pct": (counts / counts.sum() * 100).round(2),
                         "imbalance_ratio": (counts.max() / counts).round(1)}, index=CLASSES)


def majority_baseline_accuracy(y):
    """Accuracy of a model that always predicts the most common class."""
    return float(np.bincount(y).max() / len(y))


def resample(strategy, X, y, seed=42, normal_cap=5000, smote_ratio=0.2):
    """Return (X, y, sample_weight). sample_weight is None unless the strategy uses it."""
    counts = np.bincount(y)
    if strategy == "original":
        return X, y, None
    if strategy == "class_weight":
        return X, y, compute_sample_weight("balanced", y)
    if strategy == "undersample":
        Xr, yr = RandomUnderSampler(sampling_strategy={0: min(normal_cap, counts[0])}, random_state=seed).fit_resample(X, y)
        return Xr, yr, None
    if strategy in ("smote", "hybrid"):
        target = int(counts.max() * smote_ratio) if strategy == "smote" else int(normal_cap * 0.5)
        plan = {c: max(target, int(n)) for c, n in enumerate(counts) if c != 0}
        Xr, yr = SMOTE(sampling_strategy=plan, random_state=seed).fit_resample(X, y)
        if strategy == "hybrid":
            Xr, yr = RandomUnderSampler(sampling_strategy={0: min(normal_cap, counts[0])}, random_state=seed).fit_resample(Xr, yr)
        return Xr, yr, None
    raise ValueError(f"unknown strategy: {strategy}")
