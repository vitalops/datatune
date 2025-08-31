from litellm import token_counter, get_max_tokens
from typing import List


def create_batched_prompts(
    input_rows: List[str],
    batch_prefix: str,
    prompt_per_row: str,
    batch_suffix: str,
    model_name: str,
) -> List[str]:
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

    model_name = model_name[model_name.index("/") + 1 :]
    max_tokens = get_max_tokens(model_name)

    batch = ""
    batched_prompts = []
    batch_ranges = []
    nrows_per_api_call = []
    count = 0
    message = lambda x: [
        {"role": "user", "content": f"{batch_prefix}{x}{batch_suffix}"}
    ]
    prefix_suffix_tokens = token_counter(model_name, messages=message(""))
    total_ntokens = prefix_suffix_tokens

    for i, prompt in enumerate(input_rows):
        q = f"{prompt_per_row}\n {prompt} <endofrow>\n"
        batch += q
        ntokens = token_counter(model_name, messages=[{"role": "user", "content": q}])
        if total_ntokens + ntokens < max_tokens:
            count += 1
            total_ntokens += ntokens
        else:
            batch = batch[: -len(q)]
            batched_prompts.append(message(batch)[0]["content"])
            batch_ranges.append(i)
            nrows_per_api_call.append(count)

            count = 1
            batch = q
            total_ntokens = ntokens + prefix_suffix_tokens

    if count > 0:
        batched_prompts.append(message(batch)[0]["content"])
        batch_ranges.append(len(input_rows))
        nrows_per_api_call.append(count)
    return batched_prompts, batch_ranges
