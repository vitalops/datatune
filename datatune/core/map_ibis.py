import os
import ibis
import pandas as pd
from typing import Callable, List, Optional
from ibis.expr.types import Table
import pyarrow as pa
import ast
import json

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
    expected_new_fields: list[str] = []
):
    import ast, json, os
    import pandas as pd
    import pyarrow as pa
    import ibis

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

    # Row-level cache: stores JSON strings keyed by row input
    llm_cache: dict[str, str] = {}

    def run_llm_batch(input_list: list) -> list[str]:
        # Collect rows that haven't been cached yet
        to_call = []
        indices = []
        for i, val in enumerate(input_list):
            key = str(val)
            if key not in llm_cache:
                to_call.append(val)
                indices.append(i)

        # Call LLM in one batch for uncached rows
        if to_call:
            results = llm(to_call, prefix, mapping_prompt, suffix, optimized=True)
            for key_val, res in zip(to_call, results):
                try:
                    py_dict = ast.literal_eval(str(res).strip())
                    llm_cache[str(key_val)] = json.dumps(py_dict)
                except Exception:
                    llm_cache[str(key_val)] = "{}"

        # Return cached results for all rows
        return [llm_cache[str(val)] for val in input_list]

    backend_name = ibis.get_backend(table).name

    if backend_name == "duckdb":
        @ibis.udf.scalar.pyarrow
        def run_llm_batch_udf(arrow_array: str) -> str:
            input_list = arrow_array.to_pylist()
            return pa.array(run_llm_batch(input_list))
    else:
        @ibis.udf.scalar.pandas
        def run_llm_batch_udf(series: pd.Series) -> pd.Series:
            return pd.Series(run_llm_batch(series.tolist()))

    return table.mutate(**{output_col: run_llm_batch_udf(table[input_col])})



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
        self.llm_output_column = "MAP_LLM_OUTPUT__DATATUNE__"
        self.serialized_input_column = "MAP_SERIALIZED_INPUT__DATATUNE__"


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
            expected_new_fields=self.output_fields
        )

        table = apply_llm_updates(
            table,
            llm_output_col=self.llm_output_column,
            output_fields=self.output_fields
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