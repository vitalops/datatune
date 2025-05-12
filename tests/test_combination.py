from datatune.core.map import Map
from datatune.core.filter import Filter
from datatune.llm.llm import Azure
import pandas as pd
import dask.dataframe as dd
import os


def test_map_and_filter():
    # Get environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("AZURE_API_BASE")
    api_version = os.getenv("AZURE_API_VERSION")
    
    # Load test data
    df = dd.read_csv("tests/test_data/test_map.csv")
    
    # Create LLM instance
    llm = Azure(
        model_name="gpt-35-turbo",
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
    )
    

    map_prompt = "Calculate the length of each name"
    map_op = Map(prompt=map_prompt, output_fields=["first_name_length", "last_name_length"], debug=True)
    mapped_df = map_op(llm, df)
    
  
    filter_prompt = "Keep only rows where the first name length is greater than 5"
    filter_op = Filter(prompt=filter_prompt, debug=True)
    filtered_df = filter_op(llm, mapped_df)
    
    # Get results
    result = filtered_df.head(10)
    print(result)
    
    # Save the final output
    result.to_csv("tests/test_data/test_map_and_filter_results.csv", index=False)