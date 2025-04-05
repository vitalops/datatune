from datatune.core.filter import Filter
from datatune.llm.llm import Ollama


def test_filter_ollama_true():
    llm = Ollama()
    prompt = "a > b?"
    inputs = {
        "a": 5,
        "b": 2
    }
    result = Filter(prompt=prompt, input_fields=["a", "b"]).execute(llm, inputs)
    assert result is True, f"Expected True, but got {result}"


def test_filter_ollama_false():
    llm = Ollama()
    prompt = "a > b?"
    inputs = {
        "a": 1,
        "b": 2
    }
    result = Filter(prompt=prompt, input_fields=["a", "b"]).execute(llm, inputs)
    assert result is False, f"Expected False, but got {result}"
