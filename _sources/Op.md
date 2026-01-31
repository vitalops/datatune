# Op

The `Op` module provides the base operation class and utility functions for Datatune's data transformation pipeline.

## Op Class

The `Op` class serves as the abstract base class for all operations in Datatune.

### Usage

```python
from datatune.core.op import Op

class CustomOperation(Op):
    def __init__(self, name=None):
        super().__init__(name=name)
        
    def __call__(self, llm, df, *args, **kwargs):
        # Implementation of the operation
        return processed_df
```

### Parameters

- **name** (str, optional): Custom name for the operation. If None, a name is auto-generated based on the class name.

## finalize Method

The `finalize` method cleans up DataFrames after operations have been applied by applying these steps:

1. Removes rows marked for deletion
2. Removes internal columns (with "__DATATUNE__")
3. Removes error and deletion marker columns

### Usage

```python
import datatune as dt
from datatune.llm.llm import LLM
import dask.dataframe as dd

# Process data
df = dd.read_csv("data.csv")
processed_df = dt.map(prompt="Extract data")(llm, df)

# Clean up the result
result = dt.finalize(processed_df)
```

### Parameters

- **df** (Union[pd.DataFrame, dd.DataFrame]): DataFrame to finalize.
- **keep_errored_rows** (bool, optional): Whether to keep rows with errors. Defaults to False.

