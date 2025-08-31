# Agents

Datatune `Agent` allows large language models (LLMs) to autonomously plan and execute data transformation steps using natural language prompts.

## Basic Usage

```python
import datatune as dt
from datatune.llm.llm import OpenAI
import dask.dataframe as dd

# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo")

# Load data
df = dd.read_csv("data.csv")

# Initialize Agent
agent = dt.Agent(llm)

# Transform data with natural language prompt
prompt = "your prompt for data transfromation"
df = agent.do(prompt,df)
 

# Compute DataFrame
result.compute().to_csv("transformed_data.csv")
```

## Parameters

- `llm` (LLM, *required*): The large language model backend to be used for data tranformations (e.g. OpenAI, Azure etc)


## Methods
## **Agent.do()**

  `do(prompt: str, df: dask.dataframe.DataFrame) -> dask.dataframe.DataFrame`
 - Executes a natural language prompt to transform the given dataframe.

**Parameters**  
  - `prompt` (`str`, *required*): Natural language instruction describing the desired transformation.  
  - `df` (`dask.dataframe.DataFrame`, *required*): Input dataframe to transform.  

  **Returns**  
  - `dask.dataframe.DataFrame`: A transformed dataframe, ready for `.compute()` or further processing.  

`Agent.do()` internally finalizes the resultant DataFrame and therefore can be readily computed.

## Examples

For more advanced usage, see the
[examples folder](https://github.com/vitalops/datatune/tree/main/examples).
