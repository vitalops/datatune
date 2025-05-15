from typing import Optional, Callable, Union
import dask.dataframe as dd
from collections import defaultdict
import pandas as pd

from datatune.core.constants import DELETED_COLUMN, ERRORED_COLUMN

# Counter for generating unique class instance names
cls_counts = defaultdict(int)


def get_name_from_class(cls_name: str):
    """
    Generates a unique name for a class instance based on the class name.

    Increments a counter for each class type to ensure uniqueness across
    multiple instances of the same class.

    Args:
        cls_name (str): The name of the class.

    Returns:
        str: A unique name combining the class name and an incremented index.
    """
    idx = cls_counts[cls_name] + 1
    cls_counts[cls_name] = idx
    return f"{cls_name}_{idx}"


class Op:
    """
    Abstract base class for operations on DataFrames.

    This class defines the interface for operations that can be applied to DataFrames
    in the datatune framework. All operation implementations should inherit from this class
    and implement the __call__ method.

    Attributes:
        name (str): Unique identifier for this operation instance.

    Args:
        name (Optional[str], optional): Custom name for the operation.
            If None, a name is auto-generated based on the class name. Defaults to None.
    """

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = get_name_from_class(self.__class__.__name__)
        self.name = name

    def __call__(
        self, llm: Callable, df: dd.DataFrame, *args, **kwargs
    ) -> dd.DataFrame:
        """
        Executes the operation on the provided DataFrame.

        This method must be implemented by subclasses to define the specific
        operation logic.

        Args:
            llm (Callable): Language model inference function to use in the operation.
            df (dd.DataFrame): DataFrame to operate on.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dd.DataFrame: The processed DataFrame.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement __call__ method")


def finalize(
    df: Union[pd.DataFrame, dd.DataFrame], keep_errored_rows: bool = False
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Retrieve a final DataFrame by removing internal columns and filtered rows.

    This function:
    1. Removes rows marked for deletion (if keep_errored_rows is False, removes errored rows)
    2. Removes internal columns used by datatune (columns with "__DATATUNE__")
    3. Removes error and deletion marker columns

    Args:
        df (Union[pd.DataFrame, dd.DataFrame]): DataFrame to finalize.
        keep_errored_rows (bool, optional): Whether to keep rows marked for deletion.
            Defaults to False.

    Returns:
        Union[pd.DataFrame, dd.DataFrame]: The finalized DataFrame with internal
            columns removed and filtered according to deletion flags.
    """
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
