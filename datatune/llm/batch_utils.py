from litellm import token_counter, get_max_tokens
from typing import List
def token_count_string(messages:str, model_name:str):
    messages=[{"role": "user", "content": messages}]
    return token_counter(model_name, messages=messages)

def create_batch_list(prompts: List[str], model_name: str, prefix:str):
    model_name = model_name[model_name.index("/")+1:] 
    max_tokens = get_max_tokens(model_name)
    batch_str = ""
    batch_list = []
    count = []
    counts = 0
    for i in prompts:
        if token_count_string(prefix+batch_str + f"{i} <endofquestion>\n",model_name) < max_tokens:
            counts +=1
            batch_str += f"{i} <endofquestion>\n"
        else:
            batch_list.append(batch_str)
            count.append(counts)
            
             
            counts = 1
            batch_str =f"{i} <endofquestion>\n"
    
    if batch_str:
        batch_list.append(batch_str)
        count.append(counts)

    print(count)
    print("sum: ",sum(count))
    return batch_list


