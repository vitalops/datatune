"""
Defines an abstract operation over a dataframe
"""

from typing import Optional, Callable
import dask.dataframe as dd
from collections import defaultdict


cls_counts = defaultdict(int)


def get_name_from_class(cls_name: str):
    """
    Get the name of the class without the module prefix.
    """
    idx = cls_counts[cls_name] + 1
    cls_counts[cls_name] = idx
    return f"{cls_name}_{idx}"


class Op:
    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = get_name_from_class(self.__class__.__name__)
        self.name = name

    def __call__(
        self, llm: Callable, df: dd.DataFrame, *args, **kwargs
    ) -> dd.DataFrame:
        raise NotImplementedError("Subclasses must implement __call__ method")
