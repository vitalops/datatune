import dask.dataframe as dd
import datatune as dt
from datatune.llm.llm import OpenAI
import os

test_path = os.path.dirname(__file__)

def test_agent_dask():
    csv_path = os.path.join(test_path, "test_data", "dask_only.csv")
    df = dd.read_csv(csv_path)
    llm = OpenAI(model_name="gpt-4o-mini-2024-07-18", tpm=1000000, rpm=5000)
    agent = dt.agent(llm)
    prompt = "Add a new column called ProfitMargin = (Total Profit / Total Revenue) * 100."
    df = agent.do(prompt,df)
    df["ProfitMargin"] = df["ProfitMargin"].astype("float64")
    result = dt.finalize(df)
    result.compute().to_csv("agent_dask_results.csv", index=False)
    assert result.columns[-1] == "ProfitMargin"
    assert result["ProfitMargin"].dtype == "float64"


def test_agent_datatune():
    csv_path = os.path.join(test_path, "test_data", "datatune_only.csv")
    df = dd.read_csv(csv_path)
    llm = OpenAI(model_name="gpt-4o-mini-2024-07-18", tpm=1000000, rpm=5000)
    agent = dt.agent(llm)
    prompt = "Create a new column called Category and Sub-Category based on the Industry column and only keep organizations that are in Africa."
    df = agent.do(prompt,df)
    df["Category"] = df["Category"].astype("string")
    df["Sub-Category"] = df["Sub-Category"].astype("string")
    result = dt.finalize(df)
    result.compute().to_csv("agent_datatune_results.csv", index=False)
    assert result.columns[-1] == "Sub-Category"
    assert result.columns[-2] == "Category"
    assert result["Category"].dtype == "string"
    assert result["Sub-Category"].dtype == "string"


def test_agent_combined():
    csv_path = os.path.join(test_path, "test_data", "combined.csv")
    df = dd.read_csv(csv_path)
    llm = OpenAI(model_name="gpt-4o-mini-2024-07-18", tpm=1000000, rpm=5000)
    agent = dt.agent(llm)
    prompt = "Extract year from date of birth column into a new column called Year and keep only people who are in STEM related jobs."
    df = agent.do(prompt,df)
    df["Year"] = df["Year"].astype("int64")
    result = dt.finalize(df)
    result.compute().to_csv("agent_combined_results.csv", index=False)
    assert result.columns[-1] == "Year"
    assert result["Year"].dtype == "int64"
    assert len(result) <= len(df)
