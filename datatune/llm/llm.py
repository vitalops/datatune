from typing import List, Union

class LLM:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        if "temperature" not in kwargs:
            self.kwargs["temperature"] = 0.0

    def _completion(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        from litellm import completion
        response = completion(
            model=self.model_name,
            messages=messages,
            **self.kwargs
        )
        return response["choices"][0]["message"]["content"]
    
    def _batch_completion(self, prompts: List[str]) -> List[str]:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        from litellm import batch_completion
        responses = batch_completion(
            model=self.model_name,
            messages=messages,
            **self.kwargs
        )
        
        return [response["choices"][0]["message"]["content"] for response in responses]

    def __call__(self, prompt: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(prompt, str):
            return self._completion(prompt)
        return self._batch_completion(prompt)


class Ollama(LLM):
    def __init__(self, model_name="gemma3:4b", api_base="http://localhost:11434", **kwargs) -> None:
        super().__init__(model_name=f"ollama_chat/{model_name}", api_base=api_base, **kwargs)
        self.api_base = api_base
        self._model_name = model_name
