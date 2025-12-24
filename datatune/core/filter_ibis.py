import os
import ibis
import pandas as pd
from typing import Callable, List, Optional
from ibis.expr.types import Table
import pyarrow as pa
import ast
import json
import traceback

def add_serialized_col(table, target_col:str, input_fields: Optional[List[str]] = None):
    columns_to_serialize = {
        name: table[name] 
        for name in table.columns 
        if name != target_col and (input_fields is None or name in input_fields)
    }
    serialized_expr = ibis.struct(columns_to_serialize).cast("json").cast("string")
    table = table.mutate(**{target_col: serialized_expr})

    return table

def llm_batch_inference(
    table,
    llm: Callable, 
    input_col: str, 
    output_col: str,  
    prompt: str,
):
    prefix = (
        f"You are filtering a dataset. Your task is to determine whether each data record should be KEPT or REMOVED based on the filtering criteria below.{os.linesep}"
        f"Return the entire input data record with an added key called '__filter__' with value either True to KEEP the record or False to REMOVE it.{os.linesep}{os.linesep}"
    )
    filtering_prompt = f"FILTERING CRITERIA:{os.linesep}{prompt}{os.linesep}{os.linesep}"
    suffix = (
        f"{os.linesep}{os.linesep}"
        "DECISION:Your response MUST be the entire input record as  Python dictionary in the format: index=<row_index>|{key1: value1, key2: value2, ...}<endofrow> with added key called '__filter__' with value either True to KEEP the record or False to REMOVE it."
        "No explanations or additional text."
        "ALWAYS STICK TO THE FORMAT index=<row_index>|{key1: value1, key2: value2, ...}<endofrow> with added key called '__filter__' with value either True to KEEP the record or False to REMOVE it.\n"
        "IF A VALUE FOR A COLUMN DOES NOT EXIST SET IT TO None"
    )

    backend_name = ibis.get_backend(table).name

    if backend_name == "duckdb":
        @ibis.udf.scalar.pyarrow
        def run_llm_batch(arrow_array: str) -> str:
            input_list = arrow_array.to_pylist()
            results = llm(input_list, prefix, filtering_prompt, suffix, optimized=True)

            out = []
            for res in results:
                try:
                    d = ast.literal_eval(str(res).strip())
                    
                    if d:
                        last_key = list(d.keys())[-1]
                        flag = d[last_key]
                        out.append("true" if str(flag).lower() == "true" else "false")
                    else:
                        out.append("false")
                except Exception:
                    out.append("false")

            return pa.array(out)

    else:
        @ibis.udf.scalar.pandas
        def run_llm_batch(series: pd.Series) -> pd.Series:
            input_list = series.tolist()
            results = llm(input_list, prefix, filtering_prompt, suffix, optimized=True)

            out = []
            for res in results:
                try:
                    d = ast.literal_eval(str(res).strip())
                    if d:
                        last_key = list(d.keys())[-1]
                        flag = d[last_key]
                        out.append("true" if str(flag).lower() == "true" else "false")
                    else:
                        out.append("false")
                except Exception:
                    out.append("false")

            return pd.Series(out)


    return table.mutate(**{output_col: run_llm_batch(table[input_col])})

def apply_llm_filter(table, llm_output_col):
    is_true = table[llm_output_col].cast("string").lower() == "true"
    return table.filter(is_true.fill_null(False))


class filter_Ibis:
    """
    Class to handle filtering operations using Ibis tables.
    """

    def __init__(
        self,
        prompt: str,
        input_fields: Optional[List[str]] = None,
    ):
        self.prompt = prompt
        self.input_fields = input_fields or []
        self.llm_output_column = "FILTER_LLM_OUTPUT__DATATUNE__"
        self.serialized_input_column = "FILTER_SERIALIZED_INPUT__DATATUNE__"

    def __call__(self, llm: Callable, table: Table) -> Table:

        self.llm = llm

        table = add_serialized_col(
            table, 
            target_col=self.serialized_input_column, 
            input_fields=self.input_fields
        )

        table = llm_batch_inference(
            table,
            llm=self.llm,
            input_col=self.serialized_input_column,
            output_col=self.llm_output_column,
            prompt=self.prompt,
        )

        table = apply_llm_filter(
            table,
            llm_output_col=self.llm_output_column
        )
        table = table.select(
    *[
        c for c in table.schema().names
        if c not in {
            self.llm_output_column,
            self.serialized_input_column,
        }
    ]
)

        return table