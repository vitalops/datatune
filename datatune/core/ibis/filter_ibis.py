import os
import ibis
import pandas as pd
from typing import Callable, List, Optional
from ibis.expr.types import Table
import pyarrow as pa
import ast
import json
import traceback

import ibis

def add_serialized_col(table, target_col: str, input_fields: Optional[List[str]] = []):
    cols = [
        name for name in table.columns 
        if name != target_col and (not input_fields or name in input_fields)
    ]
    
    parts = []
    for i, name in enumerate(cols):
        part = ibis.literal(f'"{name}": ') + table[name].cast("string")
        parts.append(part)
    
    inner_string = ibis.literal(", ").join(parts)
    json_string_expr = ibis.literal("{") + inner_string + ibis.literal("}")
    
    return table.mutate(**{target_col: json_string_expr})

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

    indexed_table = table.mutate(_ROW_ID_=ibis.row_number().cast("int64"))

    local_data = indexed_table.select("_ROW_ID_", input_col).execute()
    
    input_list = local_data[input_col].tolist()
    
    results = llm(input_list, prefix, filtering_prompt, suffix, optimized=True)

    out_bools = []
    for res in results:
        try:
            d = ast.literal_eval(str(res).strip())
            if d:
                last_key = list(d.keys())[-1]
                flag = d[last_key]
                out_bools.append(True if str(flag).lower() == "true" else False)
            else:
                out_bools.append(False)
        except Exception:
            out_bools.append(False)

    filter_df = pd.DataFrame({
        "_ROW_ID_": local_data["_ROW_ID_"].values,
        output_col: out_bools
    })
    filter_memtable = ibis.memtable(filter_df)

    joined = indexed_table.join(
        filter_memtable, 
        indexed_table["_ROW_ID_"] == filter_memtable["_ROW_ID_"]
    )
    
    final_cols = list(table.columns) + [output_col]
    
    return joined.select(final_cols)

def apply_llm_filter(table, llm_output_col):
    return table.filter(table[llm_output_col].fill_null(False))


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
        table = ibis.memtable(table.execute())

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
             "ROW_ID_",
            "_ROW_ID__right"
        }
    ]
)

        return table