import os

import dask.dataframe as dd
import pandas as pd

import datatune as dt
from datatune.llm.llm import Azure


def test_map_and_filter():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_API_BASE")
    api_version = os.getenv("AZURE_API_VERSION")

    df = dd.read_csv("tests/test_data/test_map.csv")

    llm = Azure(
        model_name="azure/gpt-35-turbo",
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
    )

    map_prompt = "Calculate the length of each name"
    map_op = dt.map(
        prompt=map_prompt,
        output_fields=["first_name_length", "last_name_length"],
    )
    mapped_df = map_op(llm, df)

    filter_prompt = "Keep only rows where the first name length is greater than 5"
    filter_op = dt.filter(prompt=filter_prompt)
    filtered_df = filter_op(llm, mapped_df)

    result = filtered_df.head(10)

    result.to_csv("tests/test_data/test_map_and_filter_results.csv", index=False)
