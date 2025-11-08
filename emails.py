import json
import dask.dataframe as dd
import datatune as dt
from datatune.llm.llm import OpenAI
from typing import List
import os
import time


def get_actionable_items(json_email_filepath: str, llm) -> List[str]:
    with open(json_email_filepath) as f:
        data = [str(json.loads(line)) + "return a python list of actionable items for this email" for line in f]

    batch_prefix = "Response must be a python list of actionable items for each email"
    batch_suffix = "MUST ONLY RETURN VALID PYTHON LISTS OF ACTIONABLE steps FOR EACH EMAIL."
    actionable_items = llm(data,batch_prefix,batch_suffix,optimized = True,max_retries=3)
    return actionable_items
def summarize(actionable_items: List[List[str]], llm, batch_size: int = 5)-> List[str]:
    batches = [actionable_items[i:i + batch_size] for i in range(0, len(actionable_items), batch_size)]
    input_list = ["[" + ", ".join(batch) + "]" +"return a string summary of this batch of actionable items" for batch in batches]
    #time.sleep(100000)
    batch_prefix = "return a string summary of the following actionable items for each batches"
    batch_summaries = llm(input_list,batch_prefix,optimized = True)
    return batch_summaries

llm = OpenAI(model_name="gpt-4o-mini-2024-07-18",tpm=1000000,rpm=5000)
actionable_items = get_actionable_items("emails.json",llm)
summarized_items = summarize(actionable_items,llm)
print(summarized_items[0])
print(summarized_items[1])