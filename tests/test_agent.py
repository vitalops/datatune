import dask.dataframe as dd
import datatune as dt
from datatune.llm.llm import Azure
import os

test_path = os.path.dirname(__file__)
api_key = os.getenv("AZURE_API_KEY")
api_base = os.getenv("AZURE_API_BASE")
api_version = os.getenv("AZURE_API_VERSION")

def test_agent_dask():
    csv_path = os.path.join(test_path, "test_data", "dask_only.csv")
    df = dd.read_csv(csv_path)
    llm = Azure(
        model_name="gpt-4o-mini",
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
    )
    agent = dt.Agent(llm)
    prompt = (
        "Add a new column called ProfitMargin = (Total Profit / Total Revenue) * 100."
    )
    df = agent.do(prompt, df)
    df["ProfitMargin"] = df["ProfitMargin"].astype("float64")
    df.to_csv("agent_dask_results.csv", index=False)
    assert df.columns[-1] == "ProfitMargin"
    assert df["ProfitMargin"].dtype == "float64"


def test_agent_datatune():
    csv_path = os.path.join(test_path, "test_data", "datatune_only.csv")
    df = dd.read_csv(csv_path)
    llm = Azure(
        model_name="gpt-4o-mini",
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
    )
    agent = dt.Agent(llm)
    prompt = "Create a new column called Category and Sub-Category based on the Industry column and only keep organizations that are in Africa."
    df = agent.do(prompt, df)
    df["Category"] = df["Category"].astype("string")
    df["Sub-Category"] = df["Sub-Category"].astype("string")
    df.to_csv("agent_datatune_results.csv", index=False)
    assert df.columns[-1] == "Sub-Category"
    assert df.columns[-2] == "Category"
    assert df["Category"].dtype == "string"
    assert df["Sub-Category"].dtype == "string"


def test_agent_combined():
    csv_path = os.path.join(test_path, "test_data", "combined.csv")
    df = dd.read_csv(csv_path)
    llm = Azure(
        model_name="gpt-4o-mini",
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
    )
    agent = dt.Agent(llm)
    prompt = "Extract year from date of birth column into a new column called Year and keep only people who are in STEM related jobs."
    df = agent.do(prompt, df)
    df["Year"] = df["Year"].astype("int64")
    df.to_csv("agent_combined_results.csv", index=False)
    assert df.columns[-1] == "Year"
    assert df["Year"].dtype == "int64"
    assert len(df) <= len(df)
