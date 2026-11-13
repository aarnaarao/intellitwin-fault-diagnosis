from faultdiag.data import load_data
from faultdiag.features import build_features

FORBIDDEN = ["shutdown_event", "overload_event", "pressure_drop_event", "rul_days", "health_index"]


def test_no_missing_values_and_no_label_columns():
    _, X = build_features(load_data())
    assert not X.isna().any().any()
    assert not any(c in X.columns for c in FORBIDDEN)


def test_row_count_preserved():
    df = load_data()
    out, X = build_features(df)
    assert len(out) == len(X) == len(df)


def test_features_use_only_the_past():
    df = load_data()
    pump = df[df.asset_id == df.asset_id.iloc[0]].copy()
    _, X1 = build_features(pump)
    changed = pump.copy()
    changed.loc[changed.index[-1], "pressure_psi"] *= 10   # alter ONLY the last (future) reading
    _, X2 = build_features(changed)
    # every earlier row must be identical, in every feature
    assert X1.iloc[:-1].equals(X2.iloc[:-1])


def test_columns_can_be_reused():
    df = load_data()
    _, X = build_features(df)
    _, X_small = build_features(df.head(100), columns=list(X.columns))
    assert list(X_small.columns) == list(X.columns)