import pandas as pd

from datatune.data.pd import PandasDataset


def test_dataset_indexing():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [6, 7, 8, 9, 10],
            "c": [11, 12, 13, 14, 15],
        }
    )
    ds = PandasDataset(df)
    assert len(ds) == 5
    assert ds.slice == slice(None)
    assert list(ds.columns) == ["a", "b", "c"]

    idxs = [
        1,
        -2,
        slice(0, 2),
        slice(1, 4),
        slice(1, None, 2),
        slice(-1, None),
        slice(None, -1),
        slice(-3, -1),
        slice(-3, None),
    ]
    columns = [
        "b",
        ["a"],
        ["b", "c"],
        ["a", "c"],
        ["b"],
        ["a", "b"],
        ["b", "c"],
        ["a", "b", "c"],
    ]
    for idx in idxs:
        for cols in columns:
            ds2 = ds[idx][cols]
            pd_idx = idx
            if isinstance(idx, int):
                pd_idx = slice(idx, idx + 1)
            pd_cols = cols
            if isinstance(cols, str):
                pd_cols = [cols]
            df1 = df[pd_idx][pd_cols]
            assert len(ds2) == len(df1)
            assert ds2.slice == idx
            if isinstance(cols, str):
                cols = [cols]
            assert list(ds2.columns) == cols
            df2 = ds2.realize()
            assert df1.equals(df2)
