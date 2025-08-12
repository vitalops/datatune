# Map

The `Map` operation in Datatune uses LLMs to transform data in a dataset by generating new fields or modifying existing ones based on natural language instructions.

## Basic Usage

```python
from datatune.core.map import Map
from datatune.llm.llm import LLM
import dask.dataframe as dd

# Initialize LLM
llm = LLM(model_name="openai/gpt-3.5-turbo")

# Load data
df = dd.read_csv("data.csv")

# Apply transformation
mapped_df = Map(
    prompt="Extract country and city from the address field",
    output_fields=["country", "city"]
)(llm, df)

# Process results
mapped_df.compute()
```

## Parameters

- **prompt** (str, required): Natural language prompt describing the desired transformation.

- **input_fields** (List, optional): Specific fields to include in processing. If None, all fields are used.

- **output_fields** (List, optional): Names of new or modified fields to be created. Helps the LLM understand the expected output structure.

- **name** (str, optional): Custom name for the mapping operation.

## Metadata

The Map operation maintains internal columns (prefixed with the map name and suffixed with `__DATATUNE__`):

- `{name}_SERIALIZED_INPUT__DATATUNE__`: String representation of each row
- `{name}_MAP_PROMPT__DATATUNE__`: Complete prompt sent to the LLM
- `{name}_LLM_OUTPUT__DATATUNE__`: Raw LLM response

Errored rows are captured in the `ERRORED_COLUMN` as boolean.