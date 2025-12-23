import os
import ibis
import pandas as pd
from typing import Callable, List, Optional
from ibis.expr.types import Table
import pyarrow as pa
import ast
import json

def add_serialized_col(table, target_col:str="serialized_input_column", input_fields: Optional[List[str]] = None):
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
    expected_new_fields: List[str] = []
):
    prefix = (
        f"Map and transform the input according to the mapping criteria below.{os.linesep}"
        f"Replace or Create new fields or values as per the prompt. "
        f"{f'Expected new fields: {expected_new_fields}.' if expected_new_fields else ''}"
    )
    mapping_prompt = f"MAPPING CRITERIA:{os.linesep}{prompt}{os.linesep}{os.linesep}"
    suffix = (
        f"{os.linesep}{os.linesep}"
        "Your response MUST be the entire input record as a valid Python dictionary in the format "
        "'index=<row_index>|{key1: value1, key2: value2, ...}'"
    )

    backend_name = ibis.get_backend(table).name

    if backend_name == "duckdb":
        @ibis.udf.scalar.pyarrow
        def run_llm_batch(arrow_array: str) -> str:
            input_list = arrow_array.to_pylist()
            results = llm(input_list, prefix, mapping_prompt, suffix, optimized=True)
            
            cleaned_results = []
            for res in results:
                try:
                    py_dict = ast.literal_eval(str(res).strip())
                    cleaned_results.append(json.dumps(py_dict))
                except Exception:
                    cleaned_results.append("{}")
            
            return pa.array(cleaned_results)
    else:
        @ibis.udf.scalar.pandas
        def run_llm_batch(series: pd.Series) -> pd.Series:
            input_list = series.tolist()
            results = llm(input_list, prefix, mapping_prompt, suffix, optimized=True)
            
            cleaned = []
            for res in results:
                try:
                    cleaned.append(json.dumps(ast.literal_eval(str(res).strip())))
                except:
                    cleaned.append("{}")
            return pd.Series(cleaned)

    return table.mutate(**{output_col: run_llm_batch(table[input_col])})

def apply_llm_updates(table: Table, llm_output_col: str, output_fields: list) -> Table:
    all_targets = set(table.columns) | set(output_fields)
    updates = {}

    for field in all_targets:
        if field == llm_output_col:
            continue
            
        raw_val = table[llm_output_col].cast("json")[field]
        
        if field in table.columns:
            target_type = table.schema()[field]
            
            typed_val = raw_val.cast(target_type)
            updates[field] = typed_val.fill_null(table[field])
        else:
            updates[field] = raw_val.cast("string")

    return table.mutate(**updates)


class map_Ibis:
    """
    Class to handle mapping operations using Ibis tables.
    """

    def __init__(
        self,
        prompt: str,
        input_fields: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
    ):
        self.prompt = prompt
        self.input_fields = input_fields or []
        self.output_fields = output_fields or []

    def __call__(self, llm: Callable, table: Table) -> Table:

        self.llm = llm

        table = add_serialized_col(
            table, 
            target_col="serialized_input_column", 
            input_fields=self.input_fields
        )

        table = llm_batch_inference(
            table,
            llm=self.llm,
            input_col="serialized_input_column",
            output_col="llm_output_column",
            prompt=self.prompt,
            expected_new_fields=self.output_fields
        )

        table = apply_llm_updates(
            table,
            llm_output_col="llm_output_column",
            output_fields=self.output_fields
        )

        existing_cols = [c for c in table.columns 
                         if c not in self.output_fields 
                         and c not in ["serialized_input_column", "llm_output_column"]]
        
        table = table.select(*existing_cols, *self.output_fields)

        return table