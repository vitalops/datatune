import numpy as np
from dataclasses import dataclass
import abc
from typing import Dict
from copy import deepcopy
from datatune.util.indexing import (
    apply_slice,
    ROW_INDEX_TYPE,
    INDEX_TYPE,
    parse_row_and_column_indices,
)


@dataclass
class Column(abc.ABC):
    name: str
    dtype: np.dtype


class Dataset(abc.ABC):

    def __init__(self):
        self.columns: Dict[str, Column] = {}
        self.slice: ROW_INDEX_TYPE = slice(None)

    def copy(self) -> "Dataset":
        return deepcopy(self)

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, item: INDEX_TYPE) -> "Dataset":
        ret = self.copy()
        row_idx, col_idx = parse_row_and_column_indices(item)
        ret.slice = apply_slice(row_idx, ret.slice, len(self))
        if col_idx is not None:
            ret.columns = {col: self.columns[col] for col in col_idx}
        return ret

    @abc.abstractmethod
    def realize(self):
        pass
