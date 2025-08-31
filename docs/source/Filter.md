# Filter

The `Filter` operation in Datatune uses LLMs to evaluate and filter rows in a dataset based on natural language criteria.

## Basic Usage

```python
import datatune as dt
from datatune.llm.llm import LLM
import dask.dataframe as dd

# Initialize LLM
llm = LLM(model_name="openai/gpt-3.5-turbo")

# Load data
df = dd.read_csv("data.csv")

# Apply filter
filtered_df = dt.filter(
    prompt="Keep only rows where the product price is reasonable for its category"
)(llm, df)

# Finalize to remove deleted rows
result = dt.finalize(filtered_df)
result.compute()
```

## Parameters

- **prompt** (str, required): Natural language prompt specifying the filtering criteria.

- **input_fields** (List, optional): Specific fields to include in filtering. If None, all fields are used.

- **name** (str, optional): Custom name for the filter operation.

- **on_error** (str, optional): Strategy for handling LLM errors:
  - `"keep"` (default): Keep error rows, delete FALSE rows
  - `"delete"`: Delete both error rows and FALSE rows


## Metadata

The Filter operation maintains internal columns (prefixed with the filter name and suffixed with `__DATATUNE__`):

- `{name}_SERIALIZED_INPUT__DATATUNE__`: String representation of each row
- `{name}_FILTER_PROMPT__DATATUNE__`: Complete prompt sent to the LLM
- `{name}_LLM_OUTPUT__DATATUNE__`: Raw LLM response
- `{name}_RESULT__DATATUNE__`: Parsed result (1 for TRUE, 0 for FALSE, -1 for errors)


Errored rows are captured in the `ERRORED_COLUMN` as boolean.