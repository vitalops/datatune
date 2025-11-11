import json
import dask.dataframe as dd
import datatune as dt
from datatune.llm.llm import OpenAI
from typing import List
import os
import time


def get_actionable_items(json_filepath: str, llm)-> List[str]:
    prompt = "make a column called actionable items and fill it with a list of actionable itemsn for each email"
    df = dd.read_json(json_filepath,lines=True)
    mapped = dt.map(
        prompt = prompt,
        output_fields=["actionable_items"],
        input_fields=["body"]
    )(llm, df)
    result = dt.finalize(mapped)
    return result["actionable_items"].compute().tolist()

def summarize(actionable_items: List[List[str]], llm, batch_size: int = 10)-> List[str]:
    actionable_items = [str(item) for item in actionable_items]
    batches = [actionable_items[i:i + batch_size] for i in range(0, len(actionable_items), batch_size)]
    input_list = ["[" + ", ".join(batch) + "]" +"return a string summary of this batch of actionable items" for batch in batches]
    #time.sleep(100000)
    #batch_prefix = "return a string summary of the following actionable items for each batches"
    batch_summaries = llm(input_list, optimized=True)
    return batch_summaries


llm = OpenAI(model_name="gpt-4o-mini-2024-07-18",tpm=1000000,rpm=5000)
#print(get_action("emails.json",llm))
actionable_items = get_actionable_items("emails.json",llm)
print(actionable_items)
summarized_items = summarize(actionable_items,llm)
print(summarized_items[0])
#print(summarized_items[1])