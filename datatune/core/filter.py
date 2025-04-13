from typing import Dict, List, Optional, Callable
from functools import partial
import numpy as np
import pandas as pd
import os


_SERIALIZED_INPUT = "_DATATUNE_SERIALIZED_INPUT_"
_FILTER_PROMPT = "_DATATUNE_FILTER_PROMPT_"
__LLM_OUTPUT__ = "_DATATUNE_LLM_OUTPUT_"


def input_as_string(df: pd.DataFrame) -> pd.DataFrame:
    df[_SERIALIZED_INPUT] = [str(row.to_dict()) for _, row in df.iterrows()]
    return df


def filter_prompt(prompt: str, df: pd.DataFrame) -> pd.DataFrame:
    prefix = f"SAY TRUE OR FALSE TO THE FOLLOWING PROMPT:{os.linesep}{prompt}{os.linesep}INPUT: "
    suffix = (
        f"{os.linesep}INSTRUCTIONS: OUTPUT JUST TRUE OR FALSE WITHOUT ANY OTHER TEXT" ""
    )
    df[_FILTER_PROMPT] = prefix + df[_SERIALIZED_INPUT] + suffix
    return df


def llm_inference(llm: Callable, df: pd.DataFrame) -> pd.DataFrame:
    df[__LLM_OUTPUT__] = llm(df[_FILTER_PROMPT])
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


def parse_filter_output_as_int(result_column: str, df: pd.DataFrame) -> pd.DataFrame:
    df[__LLM_OUTPUT__] = df[__LLM_OUTPUT__].str.strip().str.upper()
    df[result_column] = -1
    df.loc[df[__LLM_OUTPUT__] == "TRUE", result_column] = 1
    df.loc[df[__LLM_OUTPUT__] == "FALSE", result_column] = 0
    return df


class Filter:
    def __init__(self, prompt: str, input_fields: Optional[List] = None):
        self.prompt = prompt
        self.input_fields = input_fields

    def __call__(self, llm: Callable, df: Dict, result_column: str = "filter_result"):
        df = df.map_partitions(input_as_string)
        df = df.map_partitions(
            partial(filter_prompt, self.prompt),
        )
        llm_outputs = df.map_partitions(partial(llm_inference, llm))
        return llm_outputs.map_partitions(
            partial(parse_filter_output_as_int, result_column),
        )
