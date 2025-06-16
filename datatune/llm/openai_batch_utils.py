import json
import dask
from dask import delayed
from concurrent.futures import ThreadPoolExecutor
from typing import List
from litellm import token_counter

def generate_jsonl_file(prompts:List[str],model_name:str,input_file: str,system_prompt: str = "You are a helpful assistant.")->str:
    """
    Creates a JSONL file with OpenAI-style chat completion requests for a list of prompts.

    Args:
        prompts (List[str]): List of user prompts to be formatted into JSONL entries.
        model_name (str): Name of the language model to be used in the request.
        input_file (str): Path to the output JSONL file.
        system_prompt (str, optional): The system instruction to include in each request. Defaults to a generic assistant role.

    Returns:
        str: Path to the created JSONL file.
    """

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

def parse_jsonl(jsonl_path: str) -> List[str]:
    """
    Parses a JSONL file containing LLM response data and extracts the message content.

    Args:
        jsonl_path (str): Path to the JSONL file containing LLM responses.

    Returns:
        List[str]: A list of extracted message contents from the LLM responses.
    """

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

def generate_jsonl_batches(prompts:List[str], model_name:str, no_of_rows:int, batch_size:int)->List[str]: 
    """
    Splits a list of prompts into batches and creates a JSONL file for each batch using generate_json_file and Dask.

    Args:
        prompts (List[str]): Full list of user prompts.
        model_name (str): Model to use for all prompts.
        no_of_rows (int): Total number of prompts to consider.
        batch_size (int): Number of prompts per JSONL file.

    Returns:
        List[str]: List of paths to the created JSONL batch files.
    """

    delayed_export = delayed(generate_jsonl_file)
    results = []

    for i,start in enumerate(range(0, no_of_rows, batch_size)):
        batch_prompts = prompts[start:start+batch_size]
        task = delayed_export(batch_prompts, model_name, input_file=f"batch_{i}.jsonl")
        results.append(task)
    result = dask.compute(*results)
    return list(result)


def parse_jsonl_files(jsonl_files: List[str]) ->List[str]:
    """
    Parses multiple JSONL files in parallel and collects all extracted LLM responses.

    Args:
        jsonl_files (list[str]): List of paths to JSONL files containing LLM responses.

    Returns:
        List[str]:List of all extracted message contents from all JSONL files.
    """

    all_contents = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(parse_jsonl, jsonl_files)
        for contents in results:
            all_contents.extend(contents)
    return all_contents

def token_count_jsonl(jsonl_path:str,model_name:str)->int:
    """
    Counts the number of tokens in an input jsonl file.

    Args:
        jsonl_path (str): Path to input jsonl file.

    Returns:
        int: Number of tokens.
    """

    with open(jsonl_path,"r",encoding="utf-8") as f:
        token_count = 0
        for line in f:
            try:
                data= json.loads(line)
                token_count+= token_counter(model=model_name,messages=data["body"]["messages"])
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                continue
    return token_count    
                

                


