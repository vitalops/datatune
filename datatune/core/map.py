from typing import Dict, List, Optional, Callable
from functools import partial
import numpy as np
import pandas as pd
import os


_SERIALIZED_INPUT = "_DATATUNE_SERIALIZED_INPUT_"
_MAP_PROMPT = "_DATATUNE__PROMPT_"
__LLM_OUTPUT__ = "_DATATUNE_LLM_OUTPUT_"


def input_as_string(df: pd.DataFrame) -> pd.DataFrame:
    df[_SERIALIZED_INPUT] = [str(row.to_dict()) for _, row in df.iterrows()]
    return df


def map_prompt(prompt: str, df: pd.DataFrame) -> pd.DataFrame:
    prefix = f'''
    Map and transform the input according to the following prompt
    :{os.linesep}{prompt}{os.linesep}
    INPUT: '''
    suffix = (
        f"{os.linesep}INSTRUCTIONS:Map and transform the above input according to the above prompt" ""
    )
    df[_MAP_PROMPT] = prefix + df[_SERIALIZED_INPUT] + suffix
    return df


def llm_inference(llm: Callable, df: pd.DataFrame) -> pd.DataFrame:
    df[__LLM_OUTPUT__] = llm(df[_MAP_PROMPT])
    return df


class Map:
    def __init__(self, prompt: str, input_fields: Optional[List] = None):
        self.prompt = prompt
        self.input_fields = input_fields

    def __call__(self, llm: Callable, df: Dict):
        df = df.map_partitions(input_as_string)
        df = df.map_partitions(
            partial(map_prompt, self.prompt),
        )
        llm_outputs = df.map_partitions(partial(llm_inference, llm))

        return llm_outputs[__LLM_OUTPUT__]