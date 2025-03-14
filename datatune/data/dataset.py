import numpy as np
from dataclasses import dataclass
import abc
from typing import Iterable, List, Tuple, Union, Dict
from copy import deepcopy


row_index_type = Union[int, slice, Iterable[int]]
column_index_type = Union[str, Iterable[str]]


@dataclass
class Column(abc.ABC):
    name: str
    dtype: np.dtype


class Dataset(abc.ABC):

    def __init__(self):
        self.columns: Dict[str, Column] = {}
        self.slice: Union[slice, Iterable[int]] = slice(None)

    def copy(self) -> "Dataset":
        return deepcopy(self)

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(
        self,
        item: Union[
            row_index_type, column_index_type, Tuple[row_index_type, column_index_type]
        ],
    ) -> "Dataset":
        pass
