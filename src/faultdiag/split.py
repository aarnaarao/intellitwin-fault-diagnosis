"""Pump-level split: a pump never appears in more than one split."""
from sklearn.model_selection import GroupShuffleSplit


def split_pumps(df, seed=42):
    """Return (train_pumps, val_pumps, test_pumps) as sets, 70/15/15 by pump."""
    ids = df.asset_id.values
    tr, rest = next(GroupShuffleSplit(1, train_size=0.70, random_state=seed).split(df, groups=ids))
    va, te = next(GroupShuffleSplit(1, train_size=0.50, random_state=seed).split(df.iloc[rest], groups=ids[rest]))
    return set(ids[tr]), set(ids[rest[va]]), set(ids[rest[te]])