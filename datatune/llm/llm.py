from litellm import completion

class LLM:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = completion(
            model=self.model_name,
            messages=messages,
            **self.kwargs
        )
        return response["choices"][0]["message"]["content"]


class Ollama(LLM):
    def __init__(self, model_name="llama3.2", api_base="http://localhost:11434", **kwargs) -> None:
        super().__init__(model_name=f"ollama/{model_name}", api_base=api_base, **kwargs)
        self.api_base = api_base
