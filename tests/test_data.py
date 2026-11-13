from faultdiag.data import load_data, derive_labels, CLASSES


def test_shape_and_pumps():
    df = load_data()
    assert df.shape == (36500, 35)
    assert df.asset_id.nunique() == 50


def test_class_counts():
    y = derive_labels(load_data())
    counts = {CLASSES[i]: int((y == i).sum()) for i in range(4)}
    assert counts == {"Normal": 34363, "Overload": 342, "Pressure_drop": 1067, "Shutdown": 728}