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
agent = dt.Agent(llm,verbose=True)

# Transform data with natural language prompt
prompt = "your prompt for data transfromation"
df = agent.do(prompt,df,verbose=False)
 

# Compute DataFrame
result.compute().to_csv("transformed_data.csv")
```

## Parameters

- `llm` (LLM, *required*): The large language model backend to be used for data tranformations (e.g. OpenAI, Azure etc)
- `verbose` (bool, *optional*, default=`False`): If set to `True`, the agent will print the full generated plan, show detailed information for each transformation step and display error messages if a step fails.


## Methods
## **Agent.do()**

  `do(prompt: str, df: dask.dataframe.DataFrame) -> dask.dataframe.DataFrame`
 - Executes a natural language prompt to transform the given dataframe.

**Parameters**  
  - `prompt` (`str`, *required*): Natural language instruction describing the desired transformation.  
  - `df` (`dask.dataframe.DataFrame`, *required*): Input dataframe to transform.  
  - `verbose` (`bool`, *optional*, default=`None`): Controls logging behavior for this call.
    - If set to `True`, the agent will print the full generated plan, show detailed information for each transformation step and display error messages if a step fails.
    - If `verbose` is not provided, method uses Agent's verbose setting.

  **Returns**  
  - `pandas.DataFrame`: A transformed pandas dataframe.  

## Examples

For more advanced usage, see the
[examples folder](https://github.com/vitalops/datatune/tree/main/examples).
