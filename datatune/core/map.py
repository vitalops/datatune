import ast
import json
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
    expected_new_fields: List[str],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates the mapping prompt, prefix and suffix to be prepended and appended to a batched prompt respectively.
    The LLM is called with each row's serialized input and its response is stored in a new column.

    Args:
        llm (Callable): A callable LLM function that accepts (input_rows, prefix, prompt, suffix).
        llm_output_column (str): Name of the column to store the LLM responses.
        prompt (str): Base Map prompt
        serialized_input_column (str): Name of the column containing serialized input data per row.
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column (`llm_output_column`)
        containing the LLM's responses.
    """
    prefix = (
        f"Map and transform the input according to the mapping criteria below.{os.linesep}"
        f""" Replace or Create new fields or values as per the prompt.
        {f"Expected new fields: {expected_new_fields}." if expected_new_fields else ""}
        """
    )
    prompt = f"MAPPING CRITERIA:{os.linesep}{prompt}{os.linesep}{os.linesep}"

    suffix = (
        f"{os.linesep}{os.linesep}"
        "Your response MUST be the entire input record as a valid Python dictionary in the format"
        "'index=<row_index>|{key1: value1, key2: value2, ...}'  with added keys of expected new fields if any."
         
        "ALWAYS START YOUR RESPONSE WITH 'index=<row_index>|' WHERE <row_index> IS THE INDEX OF THE ROW." \
        "IF A VALUE FOR A COLUMN DOES NOT EXIST SET IT TO None" \
        "'index=<row_index>|{key1: None, key2: value2, ...}'"
    )

    df[llm_output_column] = llm(
        df[serialized_input_column], prefix, prompt, suffix, optimized=True
    )
    return df


def parse_llm_output(llm_output: Union[str, Exception]) -> Union[Dict, Exception]:
    """
    Parses the LLM output string into a Python dictionary.

    Args:
        llm_output (Union[str, Exception]): The raw LLM output string to parse or exception received from LLM.

    Returns:
        Union[Dict, Exception]: A dictionary if parsing succeeds, or the exception if parsing fails.
    """
    if isinstance(llm_output, Exception):
        logging.error(f"LLM Error: {llm_output}")
        return llm_output
    try:
        ret = ast.literal_eval(
            llm_output[llm_output.index("{") : llm_output.index("}") + 1]
        )

        if not isinstance(ret, dict):
            raise ValueError(f"Expected a dictionary, got {type(ret)}")
        return ret
    except (SyntaxError, ValueError) as error:
        try:
            ret = json.loads(llm_output)
            if not isinstance(ret, dict):
                raise ValueError(f"Expected a dictionary, got {type(ret)}")
            return ret
        except json.JSONDecodeError:
            return error


def update_df_with_llm_output(
    llm_output_column: str,
    df: pd.DataFrame,
    expected_fields: Optional[List[str]] = None,
    meta_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Updates the DataFrame with parsed LLM outputs, handling errors and field filtering.

    Args:
        llm_output_column (str): Name of the column containing LLM outputs.
        df (pd.DataFrame): Input DataFrame to update.
        expected_fields (Optional[List[str]], optional): List of fields to include in the output.
            If None, all fields from LLM outputs are included. Defaults to None.
        meta_columns (Optional[List[str]], optional): List of columns to keep in the final DataFrame.
            If None, all columns are kept. Defaults to None.

    Returns:
        pd.DataFrame: Updated DataFrame with LLM output values and error flags.
    """
    parsed_llm_output = df[llm_output_column].apply(parse_llm_output)
    errored_rows = parsed_llm_output.apply(lambda x: isinstance(x, Exception))
    if ERRORED_COLUMN not in df.columns:
        df[ERRORED_COLUMN] = False
    df.loc[errored_rows, ERRORED_COLUMN] = True
    not_errored_rows = ~errored_rows
    parsed_llm_output = parsed_llm_output[not_errored_rows]

    if expected_fields is not None:
        for field in expected_fields:
            if field not in df.columns:
                df[field] = None

    for i, row in parsed_llm_output.items():
        for key, value in row.items():
            if expected_fields is not None and key not in expected_fields:
                continue

            if key not in df.columns:
                df[key] = None
            df.at[i, key] = value

    if meta_columns is not None:
        df = df[meta_columns]

    return df


class map(Op):
    """
    A mapping operation that uses an LLM to transform rows in a dataset.

    The Map operator applies a prompt-based approach where each row of data
    is processed by an LLM to transform, enrich, or generate new fields based
    on the provided prompt.

    Attributes:
        prompt (str): The mapping prompt to be used with the LLM.
        input_fields (Optional[List]): List of input fields to include in mapping.
        output_fields (Optional[List]): List of output fields expected from the LLM.
        name (Optional[str]): Name of the mapping operation.
        serialized_input_column (str): Column name for serialized inputs.
        prompt_column (str): Column name for generated prompts.
        llm_output_column (str): Column name for LLM responses.

    Args:
        prompt (str): The mapping prompt to be used with the LLM.
        input_fields (Optional[List], optional): List of input fields to include. Defaults to None.
        output_fields (Optional[List], optional): List of output fields expected from the LLM. Defaults to None.
        name (Optional[str], optional): Name of the mapping operation. Defaults to None.
    """

    def __init__(
        self,
        prompt: str,
        input_fields: Optional[List] = None,
        output_fields: Optional[List] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.prompt = prompt
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.serialized_input_column = f"{self.name}_SERIALIZED_INPUT__DATATUNE__"
        self.prompt_column = f"{self.name}_MAP_PROMPT__DATATUNE__"
        self.llm_output_column = f"{self.name}_LLM_OUTPUT__DATATUNE__"

    def __call__(self, llm: Callable, df: Dict):
        """
        Applies the mapping operation to the provided DataFrame using the specified LLM.

        The mapping process involves:
        1. Serializing each row to string format
        2. Creating prompts for the LLM
        3. Running LLM inference on the prompts
        4. Parsing the results and updating the DataFrame with transformed values

        Args:
            llm (Callable): Language model inference function to use for mapping.
            df (Dict): DataFrame-like object to transform (typically a Dask DataFrame).

        Returns:
            Dict: The processed DataFrame with transformed values.
        """
        df = df.map_partitions(
            partial(
                input_as_string,
                self.serialized_input_column,
                input_fields=self.input_fields,
            )
        )

        meta_dict = df._meta.dtypes.to_dict()
        meta_dict[self.llm_output_column] = str
        df = df.map_partitions(
            partial(
                llm_batch_inference,
                llm,
                self.llm_output_column,
                self.prompt,
                self.serialized_input_column,
                self.output_fields,
            ),
            meta=meta_dict,
        )

        input_cols = list(df._meta.columns)
        output_cols = input_cols.copy()

        if self.output_fields:
            for field in self.output_fields:
                if field not in output_cols:
                    output_cols.append(field)

        if ERRORED_COLUMN not in output_cols:
            output_cols.append(ERRORED_COLUMN)

        meta = pd.DataFrame(columns=output_cols)

        result = df.map_partitions(
            partial(
                update_df_with_llm_output,
                self.llm_output_column,
                expected_fields=self.output_fields,
                meta_columns=output_cols,
            ),
            meta=meta,
        )
        return result


__all__ = [
    "map",
]
