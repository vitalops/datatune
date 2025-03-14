# A Dataset that wraps a pandas DataFrame.
# Pretty pointless, except for testing.

from datatune.data.dataset import Dataset, Column


class PandasDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.columns = {col: Column(col, df[col].dtype) for col in df.columns}
        self.base_length = len(df)

    def realize(self):
        s = self.slice
        if isinstance(s, int):
            s = slice(s, s + 1)
        ret = self.df[s][list(self.columns)].copy()
        return ret
