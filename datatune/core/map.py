from typing import Dict, List, Optional, Callable
from functools import partial
import json
import ast
import re
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
    Map and transform the input according to the following prompt.
    :{os.linesep}{prompt}{os.linesep}
    INPUT: '''
    suffix = f'''{os.linesep}INSTRUCTIONS:
    Map and transform the above input according to the above prompt.
    Replace or Create new fields or values as per the prompt.
    Your response MUST be a valid Python dictionary in the format: {{key1: value1, key2: value2, ...}}
    Format your entire response as a valid Python dictionary ONLY with no other text.
    '''
    df[_MAP_PROMPT] = prefix + df[_SERIALIZED_INPUT] + suffix
    return df


def llm_inference(llm: Callable, df: pd.DataFrame) -> pd.DataFrame:
    prompts = df[_MAP_PROMPT].tolist()
    
    if len(prompts) > 1:
        responses = llm(prompts)
    elif len(prompts) == 1:
        responses = [llm(prompts[0])]
    else:
        responses = []

    df[__LLM_OUTPUT__] = responses
    return df


def parse_llm_output(output_str):

    if isinstance(output_str, dict):
        return output_str
            
    try:
        return ast.literal_eval(output_str)
    except (SyntaxError, ValueError):
        try:
            return json.loads(output_str)
        except json.JSONDecodeError:
            return {"result": output_str}


def update_df_with_llm_output(df: pd.DataFrame, expected_fields=None) -> pd.DataFrame:
    result_df = df.copy()

    if expected_fields is None and len(df) > 0:
        row = df.iloc[0]
        llm_output = parse_llm_output(row[__LLM_OUTPUT__])
        expected_fields = list(llm_output.keys())

        if not expected_fields:
            expected_fields = ['result']
    
    for field in expected_fields or []:
        if field not in result_df.columns:
            result_df[field] = None
    
    for idx, row in df.iterrows():
        llm_output = parse_llm_output(row[__LLM_OUTPUT__])
        
        for key, value in llm_output.items():
            if key not in result_df.columns:
                if expected_fields is not None and key not in expected_fields:
                    continue
                result_df[key] = None
            result_df.at[idx, key] = value

    result_df = result_df.drop(columns=[
        col for col in [_SERIALIZED_INPUT, _MAP_PROMPT, __LLM_OUTPUT__]
        if col in result_df.columns
    ])

    return result_df


class Map:
    def __init__(self, prompt: str, input_fields: Optional[List] = None, output_fields: Optional[List] = None):
        self.prompt = prompt
        self.input_fields = input_fields
        self.output_fields = output_fields

    def __call__(self, llm: Callable, df: Dict):

        df = df.map_partitions(input_as_string)
        df = df.map_partitions(
            partial(map_prompt, self.prompt),
        )

        df_with_llm = df.map_partitions(partial(llm_inference, llm))

        if self.output_fields is None:
            expected_fields = ['result']
        else:
            expected_fields = self.output_fields

        input_cols = list(df._meta.columns)
        output_cols = [col for col in input_cols if col not in [_SERIALIZED_INPUT, _MAP_PROMPT, __LLM_OUTPUT__]]
        output_cols.extend(expected_fields)
        meta = pd.DataFrame(columns=output_cols)

        update_func = partial(update_df_with_llm_output, expected_fields=expected_fields)
        
        result_df = df_with_llm.map_partitions(update_func, meta=meta)
        
        return result_df