from litellm import token_counter, get_max_tokens
from typing import List

def create_batched_prompts(prompts: List[str], model_name: str) -> List[str]:
    """
    Groups a list of prompts into batches for LLM.

    This function concatenates prompts into batch strings, ensuring that the total
    token count for each batch does not exceed the maximum token limit for the given model.

    Args:
        prompts (List[str]): List of prompts.
        model_name (str): Name of model being used.

    Returns:
        List[str]: A list of prompt batches, each within the token limit of the model.

    """
    prefix = (
        "You will be given multiple requests. Each request will:\n"
        "- End with '<endofquestion>'\n\n"
        "You MUST respond to each request in order. For each answer:\n"
        "- End with '<endofresponse>'\n"
        "- Do NOT skip or omit any requests\n"
        "Your entire response MUST include one answer per request. Respond strictly in the format described.\n\n"
        "Questions:\n"
    )

    model_name = model_name[model_name.index("/")+1:] 
    max_tokens = get_max_tokens(model_name)

    message = [{"role": "user", "content": prefix}]
    batched_prompts = []
    nrows_per_api_call = []
    count = 0
    for prompt in prompts:
        q = f"{prompt} <endofquestion>\n"
        message[0]["content"] += q
        ntokens = token_counter(model_name, messages=message)
        if ntokens < max_tokens:
            count += 1
        else:
            message[0]["content"] = message[0]["content"][:-len(q)]
            batched_prompts.append(message[0]["content"])
            nrows_per_api_call.append(count)

            count = 1
            message[0]["content"] = f"{prefix}{prompt} <endofquestion>\n"
    
    if count > 0:
        batched_prompts.append(message[0]["content"])
        nrows_per_api_call.append(count)

    print(nrows_per_api_call)
    print("No of rows:", sum(nrows_per_api_call))
    print("No of api calls:", len(nrows_per_api_call))
    return batched_prompts