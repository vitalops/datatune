from typing import List, Optional, Union
from datatune.datatune.llm.batch_utils import create_batched_prompts
import asyncio
import time
from collections import deque
from litellm import token_counter

class LLM:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
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
    
    MAX_RPM = 60
    MAX_TPM = 200000

    def _true_batch_completion(self, prompts: List[str]) -> List[Union[str,Exception]]:
        def _send(prompt_list: List[str]):
            messages = [
            [{
                "role": "user",
                "content": f"{self.prefix}{prompt}"
            }]
            for prompt in prompt_list
            ]

            from litellm import batch_completion

            responses = batch_completion(
                model=self.model_name, messages=messages, **self.kwargs
            )

            for response in responses:
                if isinstance(response, Exception):
                    ret.append(response)
                else:
                    k= response["choices"][0]["message"]["content"].split("<endofresponse>")
                    print(k)
                    print(len(k))

                    for i in k:
                        if i.strip():
                            ret.append(i.strip()) 
        
        ret = []
        tokens = 0 
        prompt_list = []
        prompts = create_batched_prompts(prompts,self.model_name,self.prefix)
        for prompt in prompts:
            message =[{"role": "user", "content": prompt}]                 
            new_tokens = token_counter(self.model_name, messages=message)
            total_tokens = new_tokens + tokens
            if (total_tokens < self.MAX_TPM) and (len(prompt_list)+1 < self.MAX_RPM):
                prompt_list.append(prompt)
                tokens = total_tokens
            else:
                _send(prompt_list)
                time.sleep(61)
                prompt_list = [prompt]
                tokens = new_tokens
        
        if prompt_list:
            _send(prompt_list)
                
        return ret
 
    def __call__(self, prompt: Union[str, List[str]]) -> List[str]:
        """Always return a list of strings, regardless of input type"""

        self.prefix =(
            "You will be given multiple requests. Each request will:\n"
            "- Start with 'Q-[number]:'\n"
            "- End with '<endofquestion>'\n\n"
            "You MUST respond to each request in order. For each answer:\n"
            "- End with '<endofresponse>'\n"
            "- Do NOT skip or omit any requests\n"
            "Your entire response MUST include one answer per request. Respond strictly in the format described.\n\n"
            "Questions:\n"
        )
        
        if isinstance(prompt, str):
            return [self._completion(prompt)]
        return self._true_batch_completion(prompt)


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
        self.model_name = model_name if model_name.startswith("gemini/") else f"gemini/{model_name}"

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
        self.model_name = model_name if model_name.startswith("mistral/") else f"mistral/{model_name}"

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
        self.model_name = model_name if model_name.startswith("huggingface/") else f"huggingface/{model_name}"

        huggingface_params = {
            "api_key": api_key,
        }

        huggingface_params = {k: v for k, v in huggingface_params.items() if v is not None}
        kwargs.update(huggingface_params)

        super().__init__(model_name=self.model_name, **kwargs)        
