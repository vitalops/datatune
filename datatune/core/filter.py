from typing import Dict, List, Optional, Callable
from functools import partial
from datatune.core.op import Op
import pandas as pd
import os
from datatune.core.constants import DELETED_COLUMN, ERRORED_COLUMN


def input_as_string(serialized_input_column: str, df: pd.DataFrame) -> pd.DataFrame:
    df[serialized_input_column] = [str(row.to_dict()) for _, row in df.iterrows()]
    return df


def filter_prompt(
    prompt: str, prompt_column: str, serialized_input_column: str, df: pd.DataFrame
) -> pd.DataFrame:
    prefix = f"SAY TRUE OR FALSE TO THE FOLLOWING PROMPT:{os.linesep}{prompt}{os.linesep}INPUT: "
    suffix = (
        f"{os.linesep}INSTRUCTIONS: OUTPUT JUST TRUE OR FALSE WITHOUT ANY OTHER TEXT" ""
    )
    df[prompt_column] = prefix + df[serialized_input_column] + suffix
    return df


def llm_inference(
    llm: Callable, llm_output_column: str, prompt_column: str, df: pd.DataFrame
) -> pd.DataFrame:
    df[llm_output_column] = llm(df[prompt_column])
    return df


def parse_filter_output(output: str, err: bool = False) -> Optional[bool]:
    output = output.strip().upper()
    if output == "TRUE":
        return True
    elif output == "FALSE":
        return False
    elif err:
        raise ValueError(
            f"Invalid response from LLM: {output}. Expected 'TRUE' or 'FALSE'."
        )
    else:
        return None


def parse_filter_output_as_int(
    result_column: str, llm_output_column: str, df: pd.DataFrame
) -> pd.DataFrame:
    llm_output = df[llm_output_column] = df[llm_output_column].str.strip().str.upper()
    df[result_column] = -1
    df.loc[llm_output == "TRUE", result_column] = 1
    df.loc[llm_output == "FALSE", result_column] = 0
    error_rows = df[result_column] == -1
    if ERRORED_COLUMN not in df.columns:
        df[ERRORED_COLUMN] = False
    df.loc[error_rows, ERRORED_COLUMN] = True
    return df


def delete_rows(result_column: str, on_error: str, df: pd.DataFrame) -> pd.DataFrame:
    if DELETED_COLUMN not in df.columns:
        df[DELETED_COLUMN] = False
    if on_error == "delete":
        df.loc[df[result_column] != 1, DELETED_COLUMN] = True
    elif on_error == "keep":
        df.loc[df[result_column] == 0, DELETED_COLUMN] = True
    else:
        raise ValueError("on_error must be either 'keep' or 'delete'")
    return df


class Filter(Op):
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
        df = df.map_partitions(partial(input_as_string, self.serialized_input_column))
        df = df.map_partitions(
            partial(
                filter_prompt,
                self.prompt,
                self.prompt_column,
                self.serialized_input_column,
            ),
        )
        llm_outputs = df.map_partitions(
            partial(llm_inference, llm, self.llm_output_column, self.prompt_column)
        )
        results = llm_outputs.map_partitions(
            partial(
                parse_filter_output_as_int, self.result_column, self.llm_output_column
            ),
        )
        return results.map_partitions(
            partial(delete_rows, self.result_column, self.on_error),
        )


__all__ =[
    "Filter",
]