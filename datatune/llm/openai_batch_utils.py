import json
import os
import dask
from dask import delayed

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from typing import List


def export_openai_jsonl(prompts:List[str],model_name:str,input_file: str,system_prompt: str = "You are a helpful assistant."):
    

    with open(input_file, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(prompts, start=1):
            item = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model":model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000
                }
            }
            json.dump(item, f)
            f.write("\n")
    return input_file

def extract_contents_from_jsonl(jsonl_path: str) -> list[str]:
    contents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                contents.append(content)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                continue
    return contents

def multiple_jsonl_export(prompts:List[str],model_name:str,no_of_rows:int,batch_size:int): 
    delayed_export = delayed(export_openai_jsonl)
    results = []

    for i,start in enumerate(range(0, no_of_rows, batch_size)):
        batch_prompts = prompts[start:start+batch_size]
        task = delayed_export(batch_prompts, model_name, input_file=f"batch_{i}.jsonl")
        results.append(task)
    result = dask.compute(*results)
    return result


def multiple_jsonl_extract(jsonl_files: list[str]) ->list[str]:

    all_contents = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_contents_from_jsonl, jsonl_files)
        for contents in results:
            all_contents.extend(contents)
    return all_contents