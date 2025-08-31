from datatune.llm.llm import Ollama
import pandas as pd
import dask.dataframe as dd
import datatune as dt


def test_filter():
    df = dd.read_csv("tests/test_data/test_filter.csv")
    prompt = "check the statement is factually correct"
    fltr = dt.filter(prompt=prompt)
    llm = Ollama()
    filtered = fltr(
        llm,
        df,
    )
    filtered = filtered.head(10)
    print(filtered)
    filtered.to_csv("tests/test_data/test_filter_result.csv", index=False)
