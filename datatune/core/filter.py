import ast
import logging
import os
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from datatune.core.constants import DELETED_COLUMN, ERRORED_COLUMN
from datatune.core.op import Op


def input_as_string(
    serialized_input_column: str,
    df: pd.DataFrame,
    input_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Converts each row in the DataFrame to a string representation and stores it in a new column.

    Args:
        serialized_input_column (str): Name of the column to store the serialized row data.
        df (pd.DataFrame): Input DataFrame to process.
        input_fields (Optional[List], optional): List of input fields to include. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with the added serialized input column.
    """

    df_inputs = df[input_fields] if input_fields else df
    df[serialized_input_column] = [
        str(row.to_dict()) for _, row in df_inputs.iterrows()
    ]
    return df


def llm_batch_inference(
    llm: Callable,
    llm_output_column: str,
    prompt: str,
    serialized_input_column: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates the filtering prompt, prefix and suffix to be prepended and appended to a batched prompt respectively.
    The LLM is called with each row's serialized input and its response is stored in a new column.

    Args:
        llm (Callable): A callable LLM function that accepts (input_rows, prefix, prompt, suffix).
        llm_output_column (str): Name of the column to store the LLM responses.
        prompt (str): Base Filter prompt
        serialized_input_column (str): Name of the column containing serialized input data per row.
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column (`llm_output_column`)
        containing the LLM's responses.
    """
    prefix = (
        f"You are filtering a dataset. Your task is to determine whether each data record should be KEPT or REMOVED based on the filtering criteria below.{os.linesep}"
        f"Return the entire input data record with an added key called '__filter__' with value either True to KEEP the record or False to REMOVE it.{os.linesep}{os.linesep}"
    )
    prompt = f"FILTERING CRITERIA:{os.linesep}{prompt}{os.linesep}{os.linesep}"
    suffix = (
        f"{os.linesep}{os.linesep}"
        "DECISION:Your response MUST be the entire input record as  Python dictionary in the format: index=<row_index>|{key1: value1, key2: value2, ...}<endofrow> with added key called '__filter__' with value either True to KEEP the record or False to REMOVE it."
        "No explanations or additional text."
        "ALWAYS STICK TO THE FORMAT index=<row_index>|{key1: value1, key2: value2, ...}<endofrow> with added key called '__filter__' with value either True to KEEP the record or False to REMOVE it.\n"
        "IF A VALUE FOR A COLUMN DOES NOT EXIST SET IT TO None"
    )
    df[llm_output_column] = llm(
        df[serialized_input_column], prefix, prompt, suffix, optimized=True
    )
    return df


def parse_filter_output(
    output: Union[str, Exception], err: bool = True
) -> Optional[bool]:
    """
    Parses the LLM output to determine TRUE/FALSE results.

    Args:
        output (Union[str, Exception]): The raw LLM output to parse.
        err (bool, optional): If True, raise an error for invalid responses.
                              If False, return None for invalid responses. Defaults to False.

    Returns:
        Optional[bool]: True, False, or None based on the parsed output.

    Raises:
        ValueError: If err=True and the output is neither 'TRUE' nor 'FALSE'.
    """
    if isinstance(output, Exception):
        logging.error(f"LLM error: {output}")
        return None

    output_dict = ast.literal_eval(output[output.index("{") : output.index("}") + 1])
    output = output_dict.get("__filter__", None)

    if isinstance(output, bool):
        return output
    elif err:
        raise ValueError(
            f"Invalid response from LLM: {output}. Expected boolean True or False."
        )
    else:
        return None


def parse_filter_output_as_int(
    result_column: str, llm_output_column: str, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Parses LLM outputs to integer values and marks error rows.

    Args:
        result_column (str): Name of the column to store the parsed results (1 for TRUE, 0 for FALSE, -1 for errors).
        llm_output_column (str): Name of the column containing LLM outputs to parse.
        df (pd.DataFrame): Input DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with added result column and updated error flags.
    """
    llm_output = df[llm_output_column].apply(parse_filter_output)
    df.loc[:, result_column] = -1
    df.loc[llm_output == True, result_column] = 1
    df.loc[llm_output == False, result_column] = 0
    error_rows = df[result_column] == -1
    if ERRORED_COLUMN not in df.columns:
        df[ERRORED_COLUMN] = False
    df.loc[error_rows, ERRORED_COLUMN] = True
    return df


def delete_rows(result_column: str, on_error: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Marks rows for deletion based on filter results and the specified error handling strategy.

    Args:
        result_column (str): Name of the column containing parsed results.
        on_error (str): Error handling strategy. Must be either 'keep' or 'delete'.
                        'keep' - Delete rows with filter results, keep error rows.
                        'delete' - Delete both rows with filter results and error rows.
        df (pd.DataFrame): Input DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with updated deletion flags.

    Raises:
        ValueError: If on_error is neither 'keep' nor 'delete'.
    """
    if DELETED_COLUMN not in df.columns:
        df[DELETED_COLUMN] = False
    if on_error == "delete":
        df.loc[df[result_column] != 1, DELETED_COLUMN] = True
    elif on_error == "keep":
        df.loc[df[result_column] == 0, DELETED_COLUMN] = True
    else:
        raise ValueError("on_error must be either 'keep' or 'delete'")
    return df


class filter(Op):
    """
    A filtering operation that uses an LLM to determine which rows to keep or delete.

    The Filter operator applies a prompt-based filtering approach where each row of data
    is evaluated by an LLM to determine if it should be kept or removed from the dataset.

    Attributes:
        prompt (str): The filtering prompt to be used with the LLM.
        input_fields (Optional[List]): List of input fields to include in filtering.
        name (Optional[str]): Name of the filter operation.
        on_error (str): Strategy for handling error rows ('keep' or 'delete').
        serialized_input_column (str): Column name for serialized inputs.
        prompt_column (str): Column name for generated prompts.
        llm_output_column (str): Column name for LLM responses.
        result_column (str): Column name for parsed results.

    Args:
        prompt (str): The filtering prompt to be used with the LLM.
        input_fields (Optional[List], optional): List of input fields to include. Defaults to None.
        name (Optional[str], optional): Name of the filter operation. Defaults to None.
        on_error (str, optional): Strategy for handling error rows. Defaults to "keep".

    Raises:
        AssertionError: If on_error is neither 'keep' nor 'delete'.
    """

    def __init__(
        self,
        prompt: str,
        input_fields: Optional[List] = None,
        name: Optional[str] = None,
        on_error: str = "keep",
    ):
        super().__init__(name=name)
        self.prompt = prompt
        self.input_fields = input_fields
        self.serialized_input_column = f"{self.name}_SERIALIZED_INPUT__DATATUNE__"
        self.prompt_column = f"{self.name}_FILTER_PROMPT__DATATUNE__"
        self.llm_output_column = f"{self.name}_LLM_OUTPUT__DATATUNE__"
        self.result_column = f"{self.name}_RESULT__DATATUNE__"
        assert on_error in (
            "keep",
            "delete",
        ), "on_error must be either 'keep' or 'delete'"
        self.on_error = on_error

    def __call__(self, llm: Callable, df: Dict):
        """
        Applies the filter operation to the provided DataFrame using the specified LLM.

        Args:
            llm (Callable): Language model inference function to use for filtering.
            df (Dict): DataFrame-like object to filter (typically a Dask DataFrame).

        Returns:
            Dict: The processed DataFrame with filter results and deletion markers.
        """
        drop_columns = [col for col in df.columns if "__DATATUNE__" in col]

        if drop_columns:
            df = df.drop(columns=drop_columns)

        df = df.map_partitions(
            partial(
                input_as_string,
                self.serialized_input_column,
                input_fields=self.input_fields,
            )
        )
        meta_dict = df._meta.dtypes.to_dict()
        meta_dict[self.llm_output_column] = str
        llm_outputs = df.map_partitions(
            partial(
                llm_batch_inference,
                llm,
                self.llm_output_column,
                self.prompt,
                self.serialized_input_column,
            ),
            meta=meta_dict,
        )
        meta = llm_outputs._meta.copy()
        meta[self.result_column] = int
        meta[ERRORED_COLUMN] = bool
        results = llm_outputs.map_partitions(
            partial(
                parse_filter_output_as_int, self.result_column, self.llm_output_column
            ),
            meta=meta,
        )
        return results.map_partitions(
            partial(delete_rows, self.result_column, self.on_error),
        )


__all__ = [
    "filter",
]
