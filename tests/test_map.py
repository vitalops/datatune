from datatune.core.map import Map
from datatune.llm.llm import LLM
import pandas as pd
import dask.dataframe as dd
import os
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("AZURE_API_BASE")
api_version = os.getenv("AZURE_API_VERSION")


def test_map_replace():
    df = dd.read_csv("tests/test_data/test_map.csv")
    prompt = "Replace all personally identifiable terms with XX"
    map = Map(prompt=prompt)
    llm = LLM(model_name='azure/gpt-35-turbo',
           api_base=api_base,
           api_version=api_version,
           api_key=api_key)
    mapped = map(llm, df)
    mapped = mapped.head(10)
    print(mapped)
    mapped.to_csv("tests/test_data/test_map_replace_results.csv", index=False)

def test_map_create():
    df = dd.read_csv("tests/test_data/test_map.csv")
    prompt = "Calculate the length of each names"
    map = Map(prompt=prompt, output_fields=["first_name_length", "last_name_length"])
    llm = LLM(model_name='azure/gpt-35-turbo',
           api_base=api_base,
           api_version=api_version,
           api_key=api_key)
    mapped = map(llm, df)
    mapped = mapped.head(10)
    print(mapped)
    mapped.to_csv("tests/test_data/test_map_create_results.csv", index=False)
