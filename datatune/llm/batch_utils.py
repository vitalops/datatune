from litellm import token_counter, get_max_tokens
from typing import List
def token_count_string(messages:str, model_name:str):
    """
    Counts the number of tokens in a given string prompt when formatted as chat completion message.

    Args:
        messages (str): The input prompt string.
        model_name (str): Name of the model used.

    Returns:
        int: The total number of tokens used by the formatted chat message.
    """
    messages=[{"role": "user", "content": messages}]
    return token_counter(model_name, messages=messages)

def create_batch_list(prompts: List[str], model_name: str, prefix:str):
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
    for i in prompts:
        if token_count_string(prefix+batch_str + f"{i} <endofquestion>\n",model_name) < max_tokens:
            count +=1
            batch_str += f"{i} <endofquestion>\n"
        else:
            batch_list.append(batch_str)
            api_calls.append(count)
            
             
            count = 1
            batch_str =f"{i} <endofquestion>\n"
    
    if batch_str:
        batch_list.append(batch_str)
        api_calls.append(count)

    print(api_calls)
    print("sum: ",sum(api_calls))
    return batch_list


