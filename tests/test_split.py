from faultdiag.data import load_data
from faultdiag.split import split_pumps


def test_split_sizes_and_no_overlap():
    tr, va, te = split_pumps(load_data())
    assert (len(tr), len(va), len(te)) == (35, 7, 8)
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_split_is_reproducible():
    df = load_data()
    assert split_pumps(df) == split_pumps(df)