import os
import ibis
import pandas as pd
from typing import Callable, List, Optional
from ibis import Table
import ast
import json
from datatune.logger import get_logger

logger = get_logger(__name__)

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
    expected_new_fields: list[str] = []
):
    prefix = (
        f"Map and transform the input according to the mapping criteria below.{os.linesep}"
        f""" Replace or Create new fields or values as per the prompt.
        {f"Expected new fields: {expected_new_fields}." if expected_new_fields else ""}
        """
    )
    mapping_prompt = f"MAPPING CRITERIA:{os.linesep}{prompt}{os.linesep}{os.linesep}"

    suffix = (
        f"{os.linesep}{os.linesep}"
        "Your response MUST be the entire input record as a valid Python dictionary in the format"
        "'index=<row_index>|{key1: value1, key2: value2, ...}'  with added keys of expected new fields if any."
         
        "ALWAYS START YOUR RESPONSE WITH 'index=<row_index>|' WHERE <row_index> IS THE INDEX OF THE ROW." \
        "IF A VALUE FOR A COLUMN DOES NOT EXIST SET IT TO None" \
        "'index=<row_index>|{key1: None, key2: value2, ...}'"
    )

    indexed_table = table.mutate(_ROW_ID_=ibis.row_number().cast("int64"))

    local_data = indexed_table.select("_ROW_ID_", input_col).execute()
    
    input_list = local_data[input_col].tolist()
    
    raw_results = llm(input_list, prefix, mapping_prompt, suffix, optimized=True)
    
    processed_results = []
    for res in raw_results:
        try:
            py_dict = ast.literal_eval(str(res).strip())
            processed_results.append(json.dumps(py_dict))
        except Exception:
            processed_results.append("{}")

    mapping_df = pd.DataFrame({
        "_ROW_ID_": local_data["_ROW_ID_"].values, 
        output_col: processed_results
    })
    mapping_table = ibis.memtable(mapping_df)

    joined = indexed_table.join(
        mapping_table, 
        indexed_table["_ROW_ID_"] == mapping_table["_ROW_ID_"]
    )
    
    final_cols = list(table.columns) + [output_col]
    
    return joined.select(final_cols)



def apply_llm_updates(table, llm_output_col: str, output_fields: list) -> Table:
    updates = {}

    for field in output_fields:
        pattern = f"['\"]{field}['\"]:\s*['\"]?([^,'\"}}]+)['\"]?"
        
        raw_val = table[llm_output_col].re_extract(pattern, 1)
        
        if field in table.columns:
            target_type = table.schema()[field]
            typed_val = raw_val.cast(target_type)
            updates[field] = typed_val.fill_null(table[field])
        else:
            updates[field] = raw_val

    return table.mutate(**updates)


class _map_ibis:
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
        if self.input_fields:
            missing = [f for f in self.input_fields if f not in table.columns]
        if missing:
            error_msg = (
                f"[datatune] Schema mismatch: The following input_fields were not found: {missing}. "
                f"Available columns: {list(table.columns)}"
            )
            logger.error(error_msg)
            
            raise ValueError(error_msg)


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
            "ROW_ID_",
            "_ROW_ID__right"
        }
    ]
)


        return table