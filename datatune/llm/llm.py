from litellm import completion
import subprocess

class LLM:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        if "temperature" not in kwargs:
            self.kwargs["temperature"] = 0.0

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = completion(
            model=self.model_name,
            messages=messages,
            **self.kwargs
        )
        print(f"Response: {response}")
        return response["choices"][0]["message"]["content"]


class Ollama(LLM):
    def __init__(self, model_name="gemma3:4b", api_base="http://localhost:11434", **kwargs) -> None:
        super().__init__(model_name=f"ollama_chat/{model_name}", api_base=api_base, **kwargs)
        self.api_base = api_base
        self._model_name = model_name
