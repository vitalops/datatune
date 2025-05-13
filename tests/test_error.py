from datatune.core.map import Map
from datatune.llm.llm import Azure
import pandas as pd
import dask.dataframe as dd
import os
from datatune.core.constants import ERRORED_COLUMN

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("AZURE_API_BASE")
api_version = os.getenv("AZURE_API_VERSION")

def test_map_error():
    data = {
        "first_name": [
            "John", "Jane", "Alice", None, "Bob",
            "Charlie", "Jordan", "Eve", "Mallory", None
        ],
        "email": [
            "john@example.com", "jane.smith@gmail.com", "alice@yahoo.com", "bad_email", None,
            "charlie@aol.com", "invalid@", "eve@domain.com", "mallory@site.net", "noemail.com"
        ],
        "last_name": [
            "Doe", "Smith", "Johnson", "Brown", None,
            "Davis", "Miller", "Wilson", "", "Anderson"
        ]
    }


    df = dd.from_pandas(pd.DataFrame(data), npartitions=2)
    prompt = "Instead of responding as a dictionary with map, respond with something that causes syntaxerror if name starts with J"
    map = Map(prompt=prompt, debug=True)

    llm = Azure(
        model_name="gpt-35-turbo",
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
    )

    mapped = map(llm, df).compute()
    print(mapped)
    mapped.to_csv("tests/test_data/test_map_error_results.csv", index=False)
