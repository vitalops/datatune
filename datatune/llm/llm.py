import ast
import time
from typing import Dict, List, Optional, Union

from litellm import get_max_tokens, token_counter
from litellm import batch_completion
from datatune.llm.model_rate_limits import model_rate_limits
from datatune.logger import get_logger

logger = get_logger(__name__)


class LLM:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = (
            model_name  # of the form "provider/model" e.g. "openai/gpt-3.5-turbo"
        )
        self._base_model_name = model_name.split("/", 1)[1]  # e.g. "gpt-3.5-turbo"
        self.kwargs = kwargs
        DEFAULT_MODEL = "gpt-3.5-turbo"
        if self._base_model_name in model_rate_limits:
            model_limits = model_rate_limits[self._base_model_name]
        else:
            model_limits = model_rate_limits[DEFAULT_MODEL]
            if "rpm" not in kwargs:
                logger.warning(
                    f"REQUESTS-PER-MINUTE limits for model '{model_name}' not found. Defaulting to '{DEFAULT_MODEL}' limits: {model_limits['rpm']} RPM. Set limits by passing tpm,rpm arguments to your llm "
                )
            if "tpm" not in kwargs:
                logger.warning(
                    f"TOKENS-PER-MINUTE limits for model '{model_name}' not found. Defaulting to '{DEFAULT_MODEL}' limits: {model_limits['tpm']} TPM. Set limits by passing tpm,rpm arguments to your llm "
                )

        self.MAX_RPM = kwargs.get("rpm", model_limits["rpm"])
        self.MAX_TPM = kwargs.get("tpm", model_limits["tpm"])

        if "temperature" not in kwargs:
            self.kwargs["temperature"] = 0.0

    def _completion(self, prompt: str) -> Union[str, Exception]:
        messages = [{"role": "user", "content": prompt}]
        from litellm import completion

        response = completion(model=self.model_name, messages=messages, **self.kwargs)

        if isinstance(response, Exception):
            return response
        return response["choices"][0]["message"]["content"]

    def get_max_tokens(self) -> int:
        return get_max_tokens(self._base_model_name)

    def _create_batched_prompts(
        self,
        input_rows: List[str],
        batch_prefix: str,
        prompt_per_row: str,
        batch_suffix: str,
        retries: int,
    ) -> List[str]:
        """
        Groups a list of prompts into batches for LLM.

        This function concatenates prompts into batch strings, ensuring that the total
        token count for each batch does not exceed the maximum token limit for the given model.

        Args:
            input_rows (List[str]): List of input prompts.

            retries (int): Number of retries for failed rows.

        Returns:
            List[str]: A list of prompt batches, each within the token limit of the model.

        """
        batch_prefix = (
            "You will be given multiple data rows to process. Each request will:\n"
            "- have the format 'index=<row_index>|{<row_data>}' where <row_index> is the zero-based index of the row in the original input list.\n"
            "- End with '<endofrow>'\n\n"
            "You MUST respond to each row in order. Each answer:\n"
            "MUST BE OF THE FORMAT 'index=<row_index>|{response}' where <row_index> is the zero-based index of the row in the original input list.\n" \
            "{response} must be enclosed in curly braces and strings should be enclosed in quotes.\n"
            "- End with '<endofrow>'\n" \
            "Always begin your response with 'index=<row_index>|' to indicate which row you are responding to without exception.\n"
            "- Do NOT skip or omit any rows\n"
            "Your entire response MUST include one answer per row. Respond strictly in the format described WITHOUT ANY OTHER TEXT, EXPLANATIONS OR BACKSTICKS\n" \
            "IF DATA ROWS ARE NOT GIVEN IN DICTIONARY FORMAT RETURN THE ANSWER ONLY STARTING WITH index=<row_index>|"
            "ALL RESPONSES MUST START WITH 'index='\n"
            f"Instructions:\n{batch_prefix or ''}"
        )

        max_tokens = self.get_max_tokens()
        model_name = self._base_model_name

        batch = ""
        batched_prompts = []
        batch_ranges = []
        nrows_per_api_call = []
        count = 0
        message = lambda x: [
            {"role": "user", "content": f"{batch_prefix or ''}{x}{batch_suffix or ''}"}
        ]
        prefix_suffix_tokens = token_counter(model_name, messages=message(""))
        total_ntokens = prefix_suffix_tokens

        for i, prompt in enumerate(input_rows):
            q = f"{prompt_per_row or ''}\n {prompt} <endofrow>\n"
            batch += q
            ntokens = token_counter(
                model_name, messages=[{"role": "user", "content": q}]
            )
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
        if retries > 1:
            logger.info(f"🔄 Retrying failed rows\n")
        logger.info(f"📦 Prompts have been batched: {nrows_per_api_call}")
        logger.info(f"📝 Total rows to process: {sum(nrows_per_api_call)}")
        logger.info(f"📤 Number of batches to send: {len(nrows_per_api_call)}\n")

        return batched_prompts, batch_ranges

    def _batch_completion(self, prompts: List[str]) -> List[Union[str, Exception]]:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        

        responses = batch_completion(
            model=self.model_name, messages=messages, **self.kwargs
        )

        ret = []
        for response in responses:
            if isinstance(response, Exception):
                ret.append(response)
            else:
                ret.append(response["choices"][0]["message"]["content"])

        return ret
        
    def optimized_batch_completion(
        self,
        input_rows: List[str],
        batch_prefix: str,
        prompt_per_row: str,
        batch_suffix: str,
    ) -> List[Union[str, Exception]]:
        input_rows = list(input_rows)
        """
    Executes completions on batched input prompts without trigerring RateLimitErrors and retries failed requests
    by associating responses with original inputs via indexing.

    Adds an "index" key to each row to enable correct mapping of outputs.
    Responses are split using `<endofrow>` delimiters and parsed as Python dictionaries. Outputs are
    returned in the order of the original input rows.


    Args:
        input_rows (List[str]): List of input rows
        batch_prefix (str): A shared prefix prepended to each batch.
        prompt_per_row (str): Prompt to transform the input row.
        batch_suffix (str): A shared suffix appended to each batch.

    Returns:
        List[Union[str, Exception]]: A list containing the parsed LLM responses.
    """

        input_rows = [f"index={i}|{row}" for i, row in enumerate(input_rows)]

        remaining = set(range(len(input_rows)))
        ret = [None] * len(input_rows)

        def _send(messages: List[Dict[str, str]], batch_ranges: List[int]):
            """
            Sends a batch of prompts and processes the returned content.

            The function extracts the index from each result to associate it with the corresponding input.
            Valid results are stored in their original positions. Malformed or duplicate entries are skipped.

            Args:
                messages (List[Dict[str, str]]): A chat-message dictionary with a batched prompt as content.
                batch_ranges (List[int]): Each element of the list is the number of rows in each batch.

            """

            logger.info(f"📨 {len(messages)} Batches sent\n")
            logger.info(f"⏳ Waiting for responses...")
            responses = batch_completion(
                model=self.model_name, messages=messages, **self.kwargs
            )
            logger.info(f"📬 Responses received")

            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    start, end = 0 if i == 0 else batch_ranges[i - 1], batch_ranges[i]
                    for idx in range(start, end):
                        ret[idx] = response
                        remaining.remove(idx)
                else:
                    n = 0
                    for result in response["choices"][0]["message"]["content"].split(
                        "<endofrow>"
                    ):
                        result = result.strip()
                        if result:
                            try:
                                sep_idx = result.index("|")
                                idx = int(result[result.index("index=") + 6 : sep_idx])
                                result = result[sep_idx + 1 :]
                                result = result.split("{", 1)[1]
                                result = result.rsplit("}", 1)[0]
                                result = "{" + result + "}"
                                result = ast.literal_eval(result)
                                if isinstance(result, set):
                                    result = next(iter(result))
                            except:
                                continue
                            if idx not in remaining:
                                continue
                            remaining.remove(idx)
                            ret[idx] = str(result)
                            n += 1
                                     
        retries = 0
        while remaining:
            remaining_prompts = [input_rows[i] for i in remaining]
            ntokens = 0
            start = 0
            retries += 1
            messages = []
            batched_prompts, batch_ranges = self._create_batched_prompts(
                remaining_prompts,
                batch_prefix,
                prompt_per_row,
                batch_suffix,
                retries,
                
            )
            for i, batched_prompt in enumerate(batched_prompts):
                message = [{"role": "user", "content": batched_prompt}]
                curr_ntokens = token_counter(self.model_name, messages=message)
                total = curr_ntokens + ntokens
                if (total < self.MAX_TPM) and (len(messages) + 1 < self.MAX_RPM):
                    messages.append(message)
                    ntokens = total
                else:
                    t1 = time.time()
                    _send(messages, batch_ranges[start:i])
                    t2 = time.time()
                    time.sleep(max(0, 61 - (t2 - t1)))
                    messages = [message]
                    ntokens = curr_ntokens

            if messages:
                _send(messages, batch_ranges[start:])
        logger.info(f"✅ Processed {len(ret)} rows\n")
        return ret

    def __call__(
        self, prompt: Union[str, List[str]], batch_prefix: str=None,prompt_per_row: str=None, batch_suffix: str=None, optimized: bool = False
    ) -> Union[str, List[str]]:
        if isinstance(prompt, str):
            return self._completion(prompt)
        if optimized:
            return self.optimized_batch_completion(prompt, batch_prefix, prompt_per_row, batch_suffix)
        return self._batch_completion(prompt)


class Ollama(LLM):
    def __init__(
        self, model_name="gemma3:4b", api_base="http://localhost:11434", **kwargs
    ) -> None:
        super().__init__(
            model_name=f"ollama_chat/{model_name}", api_base=api_base, **kwargs
        )
        self.api_base = api_base
        self._model_name = model_name


class OpenAI(LLM):
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **kwargs):
        kwargs.update({"api_key": api_key})
        super().__init__(model_name=f"openai/{model_name}", **kwargs)


class Azure(LLM):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        azure_model = f"azure/{self.model_name}"
        azure_params = {
            "api_key": api_key,
            "api_base": api_base,
            "api_version": api_version,
        }

        azure_params = {k: v for k, v in azure_params.items() if v is not None}

        kwargs.update(azure_params)
        super().__init__(model_name=azure_model, **kwargs)


class Gemini(LLM):
    def __init__(
        self,
        model_name: str = "gemini/gemma-3-1b-it",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model_name = (
            model_name if model_name.startswith("gemini/") else f"gemini/{model_name}"
        )

        gemini_params = {
            "api_key": api_key,
        }

        gemini_params = {k: v for k, v in gemini_params.items() if v is not None}
        kwargs.update(gemini_params)

        super().__init__(model_name=self.model_name, **kwargs)


class Mistral(LLM):
    def __init__(
        self,
        model_name: str = "mistral/mistral-tiny",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model_name = (
            model_name if model_name.startswith("mistral/") else f"mistral/{model_name}"
        )

        mistral_params = {
            "api_key": api_key,
        }

        mistral_params = {k: v for k, v in mistral_params.items() if v is not None}
        kwargs.update(mistral_params)

        super().__init__(model_name=self.model_name, **kwargs)


class Huggingface(LLM):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model_name = (
            model_name
            if model_name.startswith("huggingface/")
            else f"huggingface/{model_name}"
        )

        huggingface_params = {
            "api_key": api_key,
        }

        huggingface_params = {
            k: v for k, v in huggingface_params.items() if v is not None
        }
        kwargs.update(huggingface_params)

        super().__init__(model_name=self.model_name, **kwargs)
