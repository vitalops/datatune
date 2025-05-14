"""
Defines an abstract operation over a dataframe
"""

from typing import Optional, Callable, Union
import dask.dataframe as dd
from collections import defaultdict
import pandas as pd

from datatune.core.constants import DELETED_COLUMN, ERRORED_COLUMN

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



def finalize(
    df: Union[pd.DataFrame, dd.DataFrame],
    keep_errored_rows: bool = False
) -> Union[pd.DataFrame, dd.DataFrame]:

    if not keep_errored_rows and DELETED_COLUMN in df.columns:
        if not isinstance(df, dd.DataFrame):
            df = df[~df[DELETED_COLUMN]]
        else:
            df = df[~df[DELETED_COLUMN]]

    drop_columns = [col for col in df.columns if "__DATATUNE__" in col]
    drop_columns += [ERRORED_COLUMN, DELETED_COLUMN]
    if drop_columns:
        df = df.drop(columns=drop_columns)
    
    return df