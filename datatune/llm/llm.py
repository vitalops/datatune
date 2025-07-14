import ast
from typing import Dict, List, Optional, Union
from datatune.llm.batch_utils import create_batched_prompts
import time
from litellm import token_counter
from datatune.llm.model_rate_limits import model_rate_limits
import os


class LLM:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        DEFAULT_MODEL = "gpt-3.5-turbo"
        if model_name[model_name.index("/") + 1 :] in model_rate_limits:
            model_limits = model_rate_limits[model_name[model_name.index("/") + 1 :]]
        else:
            model_limits = model_rate_limits[DEFAULT_MODEL]
            if "rpm" not in kwargs:
                print(
                    f"REQUESTS-PER-MINUTE limits for model '{model_name}' not found. Defaulting to '{DEFAULT_MODEL}' limits: {model_limits['rpm']} RPM. Set limits by passing tpm,rpm arguments to your llm "
                )
            if "tpm" not in kwargs:
                print(
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

    def _batch_completion(self, prompts: List[str]) -> List[Union[str, Exception]]:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        from litellm import batch_completion

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
    
    def get_input_fields(self, first_row:Dict, prompt:str)->List[str]:
        from litellm import completion

        prompt = (
            f"You are a column selector. Your only job is to pick the minimum required columns to fulfill the request.{os.linesep}"
            f"You do NOT fulfill or execute any request. You ONLY identify which columns are required to perform the request.{os.linesep}"
            f"Given the first row of a dataset:{os.linesep}{first_row}{os.linesep}"
            f"Task: Analyze ONLY the column names. Return ONLY the minimal set of column names whose values are absolutely required to perform this request:{os.linesep}"
            f"\"{prompt}\"{os.linesep}"
            f"If in doubt, include fewer columns.{os.linesep}"
            f"Your response must be a valid Python list ONLY, in the format ['column1', 'column2'] with no explanations, text, or code fences.{os.linesep}"
        )

        message = [{"role":"user", "content":prompt}]

        response = completion(
                    model=self.model_name, messages=message, **self.kwargs
                )
        response_str = response["choices"][0]["message"]["content"]
        input_fields = ast.literal_eval(response_str[response_str.index('['):response_str.index(']')+1])
        return input_fields

    def true_batch_completion(
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

        
        for i in range(len(input_rows)):
            input_rows[i] = input_rows[i].strip()
            assert input_rows[i][-1] == "}", input_rows[i]
            input_rows[i] = input_rows[i][:-1] + f', "__index__": {i}}}'
            

        remaining = set(range(len(input_rows)))
        ret = [None] * len(input_rows)

        def _send(messages: List[Dict[str, str]], batch_ranges:List[int]):
            """
            Sends a batch of prompts and processes the returned content.

            The function extracts the index from each result to associate it with the corresponding input.
            Valid results are stored in their original positions. Malformed or duplicate entries are skipped.

            Args: 
                messages (List[Dict[str, str]]): A chat-message dictionary with a batched prompt as content.
                batch_ranges (List[int]): Each element of the list is the number of rows in each batch.

            """
            from litellm import batch_completion

            print("Number of batches sent: ",len(messages))
            responses = batch_completion(
                model=self.model_name, messages=messages, **self.kwargs
            )

            for i,response in enumerate(responses):
                if isinstance(response, Exception):
                    start, end = 0 if i == 0 else batch_ranges[i - 1], batch_ranges[i]
                    for idx in range(start, end):
                        ret[idx] = response
                        remaining.remove(idx)
                else:
                    n = 0
                    for result in response["choices"][0]["message"]["content"].split("<endofrow>"):
                        result = result.strip()
                        if result:
                            try:
                                result = ast.literal_eval(
                                    result[result.index("{") : result.index("}") + 1]
                                )
                                idx = result.pop("__index__")
                            except:
                                continue
                            if idx not in remaining:
                                continue
                            remaining.remove(idx)
                            ret[idx] = str(result)
                            print(result)
                            n += 1
                    print(n)
                    print()

        while remaining:
            remaining_prompts = [input_rows[i] for i in remaining]
            ntokens = 0
            start=0
            messages = []
            batched_prompts,batch_ranges = create_batched_prompts(
                remaining_prompts,
                batch_prefix,
                prompt_per_row,
                batch_suffix,
                self.model_name,
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

        print(len(ret), "rows returned")
        return ret

    def __call__(self, prompt: Union[str, List[str]]) -> List[str]:
        """Always return a list of strings, regardless of input type"""
        if isinstance(prompt, str):
            return [self._completion(prompt)]
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
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
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
