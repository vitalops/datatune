import abc
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import numpy as np

from datatune.util.indexing import (
    INDEX_TYPE,
    ROW_INDEX_TYPE,
    apply_slice,
    parse_row_and_column_indices,
    slice_length,
)


@dataclass
class Column:
    name: str
    dtype: np.dtype


class Dataset(abc.ABC):
    def __init__(self):
        self.columns: Dict[str, Column] = {}
        self.slice: ROW_INDEX_TYPE = slice(None)
        self.base_length: int = 0

    def copy(self) -> "Dataset":
        return deepcopy(self)

    def __len__(self) -> int:
        return slice_length(self.slice, self.base_length)

    def __getitem__(self, item: INDEX_TYPE) -> "Dataset":
        ret = self.copy()
        row_idx, col_idx = parse_row_and_column_indices(item)
        ret.slice = apply_slice(row_idx, ret.slice, self.base_length)
        if col_idx is not None:
            if isinstance(col_idx, str):
                ret.columns = {col_idx: self.columns[col_idx]}
            else:
                ret.columns = {col: self.columns[col] for col in col_idx}
        return ret

    @abc.abstractmethod
    def realize(self):
        pass
