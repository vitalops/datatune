from typing import Dict, List, Optional, Callable, Union
from functools import partial
import json
import ast
from datatune.core.op import Op
import pandas as pd
import os
from datatune.core.constants import DELETED_COLUMN, ERRORED_COLUMN


def input_as_string(serialized_input_column: str, df: pd.DataFrame) -> pd.DataFrame:
    df[serialized_input_column] = [str(row.to_dict()) for _, row in df.iterrows()]
    return df


def map_prompt(prompt: str, prompt_column: str, serialized_input_column: str, df: pd.DataFrame) -> pd.DataFrame:
    prefix = f"""
    Map and transform the input according to the following prompt.
    :{os.linesep}{prompt}{os.linesep}
    INPUT: """
    suffix = f"""{os.linesep}INSTRUCTIONS:
    Map and transform the above input according to the above prompt.
    Replace or Create new fields or values as per the prompt.
    Your response MUST be a valid Python dictionary in the format: {{key1: value1, key2: value2, ...}}
    Format your entire response as a valid Python dictionary ONLY with no other text.
    """
    df[prompt_column] = prefix + df[serialized_input_column] + suffix
    return df


def llm_inference(
    llm: Callable, llm_output_column: str, prompt_column: str, df: pd.DataFrame
) -> pd.DataFrame:
    df[llm_output_column] = llm(df[prompt_column])
    return df


def parse_llm_output(llm_output: str) -> Union[Dict, Exception]:
    try:
        ret = ast.literal_eval(llm_output)
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
    serialized_input_column: str, 
    prompt_column: str, 
    df: pd.DataFrame, 
    expected_fields: Optional[List[str]] = None,
    meta_columns=None
) -> pd.DataFrame:
    parsed_llm_output = df[llm_output_column].apply(parse_llm_output)
    # TODO(fariz): vectorize this?
    errored_rows = parsed_llm_output.apply(
        lambda x: isinstance(x, Exception)
    )
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


class Map(Op):
    def __init__(
        self,
        prompt: str,
        input_fields: Optional[List] = None,
        output_fields: Optional[List] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.prompt = prompt
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.serialized_input_column = f"{self.name}_SERIALIZED_INPUT__DATATUNE__"
        self.prompt_column = f"{self.name}_MAP_PROMPT__DATATUNE__"
        self.llm_output_column = f"{self.name}_LLM_OUTPUT__DATATUNE__"

    def __call__(self, llm: Callable, df: Dict):
        df = df.map_partitions(partial(input_as_string, self.serialized_input_column))
        df = df.map_partitions(
            partial(
                map_prompt,
                self.prompt,
                self.prompt_column,
                self.serialized_input_column,
            ),
        )
        df = df.map_partitions(
            partial(llm_inference, llm, self.llm_output_column, self.prompt_column)
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
                self.serialized_input_column,
                self.prompt_column,
                expected_fields=self.output_fields,
                meta_columns=output_cols
            ),
            meta=meta
        )
        return result


__all__ =[
    "Map",
]