from datatune.core.map import Map
from datatune.llm.llm import Ollama


def test_map_ollama():
    llm = Ollama()
    prompt = "add a and b and return the result as c"
    inputs = {
        "a": 1,
        "b": 2
    }
    result = Map(prompt=prompt, input_fields=["a", "b"], output_fields=["c"]).execute(llm, inputs)
    assert result == {"c": 3}, f"Expected {{'c': 3}}, but got {result}"
