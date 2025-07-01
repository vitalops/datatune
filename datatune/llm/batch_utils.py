from litellm import token_counter, get_max_tokens
from typing import List

def create_batched_prompts(input_rows: List[str], batch_prefix: str, prompt_per_row: str, batch_suffix: str, model_name: str) -> List[str]:
    """
    Groups a list of prompts into batches for LLM.

    This function concatenates prompts into batch strings, ensuring that the total
    token count for each batch does not exceed the maximum token limit for the given model.

    Args:
        input_rows (List[str]): List of input prompts.
        
        model_name (str): Name of model being used.

    Returns:
        List[str]: A list of prompt batches, each within the token limit of the model.

    """
    batch_prefix = (
        "You will be given multiple data rows to process. Each request will:\n"
        "- End with '<endofrow>'\n\n"
        "You MUST respond to each row in order. For each answer:\n"
        "- End with '<endofrow>'\n"
        "- Do NOT skip or omit any rows\n"
        "Your entire response MUST include one answer per row. Respond strictly in the format described.\n\n"
        f"Instructions:\n{batch_prefix}"
    )

    model_name = model_name[model_name.index("/")+1:] 
    max_tokens = get_max_tokens(model_name)

    batch = ""
    batched_prompts = []
    nrows_per_api_call = []
    count = 0
    message = lambda x: [{"role": "user", "content": f"{batch_prefix}{x}{batch_suffix}"}]
    for prompt in input_rows:
        q = f"{prompt_per_row}\n {prompt} <endofrow>\n"
        batch += q
        ntokens = token_counter(model_name, messages=message(batch))
        if ntokens < max_tokens:
            count += 1
        else:
            batch = batch[:-len(q)]
            batched_prompts.append(message(batch)[0]["content"])
            nrows_per_api_call.append(count)

            count = 1
            batch = q
    
    if count > 0:
        batched_prompts.append(message(batch)[0]["content"])
        nrows_per_api_call.append(count)

    print(nrows_per_api_call)
    print("No of rows:", sum(nrows_per_api_call))
    print("No of api calls:", len(nrows_per_api_call))
    return batched_prompts

def get_model_limits(model_name:str):
    model_rate_limits = {
        "gpt-3.5-turbo": {
            "tpm": 200_000,
            "rpm": 500,
        },
        "gpt-3.5-turbo-0125": {
            "tpm": 200_000,
            "rpm": 500,
        },
        "gpt-3.5-turbo-1106": {
            "tpm": 200_000,
            "rpm": 500,
        },
        "gpt-3.5-turbo-16k": {
            "tpm": 200_000,
            "rpm": 500,
        },
        "gpt-3.5-turbo-instruct": {
            "tpm": 90_000,
            "rpm": 3500,
        },
        "gpt-3.5-turbo-instruct-0914": {
            "tpm": 90_000,
            "rpm": 3500,
        },
        "gpt-4": {
            "tpm": 10_000,
            "rpm": 500,
        },
        "gpt-4-0613": {
            "tpm": 10_000,
            "rpm": 500,
        },
        "gpt-4-turbo": {
            "tpm": 30_000,
            "rpm": 500,
        },
        "gpt-4-turbo-2024-04-09": {
            "tpm": 30_000,
            "rpm": 500,
        },
        "gpt-4-turbo-preview": {
            "tpm": 30_000,
            "rpm": 500,
        },
        "gpt-4-0125-preview": {
            "tpm": 30_000,
            "rpm": 500,
        },
        "gpt-4-1106-preview": {
            "tpm": 30_000,
            "rpm": 500,
        },
        "gpt-4.1": {
            "tpm": 30_000,
            "rpm": 500,
        },
        "gpt-4.1-2025-04-14": {
            "tpm": 30_000,
            "rpm": 500,
        },
        "gpt-4.1-mini": {
            "tpm": 200_000,
            "rpm": 500,
        },
        "gpt-4.1-mini-2025-04-14": {
            "tpm": 200_000,
            "rpm": 500,
        },
        "gpt-4.1-nano": {
            "tpm": 200_000,
            "rpm": 500,
        },
        "gpt-4.1-nano-2025-04-14": {
            "tpm": 200_000,
            "rpm": 500,
        },
        "gpt-4.5-preview": {
            "tpm": 125_000,
            "rpm": 1000,
        },
        "gpt-4.5-preview-2025-02-27": {
            "tpm": 125_000,
            "rpm": 1000,
        },
        "gpt-4o": {
            "tpm": 30_000,
            "rpm": 500,
        },
        "gpt-4o-2024-05-13": {
            "tpm": 30_000,
            "rpm": 500,
        },
        "gpt-4o-2024-08-06": {
            "tpm": 30_000,
            "rpm": 500,
        },
        "gpt-4o-2024-11-20": {
            "tpm": 30_000,
            "rpm": 500,
        },
    }
    model_name = model_name[model_name.index("/")+1:] 
    if model_name in model_rate_limits:
        return model_rate_limits[model_name]
    return model_rate_limits["gpt-3.5-turbo"]












