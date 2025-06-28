from litellm import token_counter, get_max_tokens
from typing import List

def create_batched_prompts(prompts: List[str], model_name: str, prefix:str):
    """
    Groups a list of prompts into batches for LLM.

    This function concatenates prompts into batch strings, ensuring that the total
    token count for each batch does not exceed the maximum token limit for the given model.

    Args:
        prompts (List[str]): List of prompts.
        model_name (str): Name of model being used.
        prefix (str): Context for each batch.

    Returns:
        List[str]: A list of prompt batches, each within the token limit of the model.

    """
    model_name = model_name[model_name.index("/")+1:] 
    max_tokens = get_max_tokens(model_name)
    batch_str = ""
    batch_list = []
    api_calls = []
    count = 0
    for prompt in prompts:
        message =[{"role": "user", "content": f"{prefix}{batch_str}Q-{count}: {prompt} <endofquestion>\n"}]                 
        total_tokens = token_counter(model_name, messages=message)
        if total_tokens < max_tokens:
            count +=1
            batch_str += f"Q-{count}: {prompt} <endofquestion>\n"
        else:
            batch_list.append(batch_str)
            api_calls.append(count)
            
             
            count = 1
            batch_str =f"Q-{count}: {prompt} <endofquestion>\n"
    
    if batch_str:
        batch_list.append(batch_str)
        api_calls.append(count)

    print(api_calls)
    print("No of rows: ",sum(api_calls))
    print("No of api calls: ",len(api_calls))
    return batch_list