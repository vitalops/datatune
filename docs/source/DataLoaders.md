# Data Loaders

Datatune supports multiple data loading backends to work with your data efficiently. Choose the backend that best fits your use case and infrastructure.

## Dask DataFrames

Datatune leverages Dask DataFrames to enable scalable processing across large datasets. This approach allows you to:

- Process data larger than the context length of LLMs
- Execute parallel computations efficiently
- Handle datasets that don't fit in memory

### Loading Data with Dask

```python
import dask.dataframe as dd

# Load from CSV
df = dd.read_csv("data.csv")

# Load from Parquet
df = dd.read_parquet("data.parquet")

# Load from multiple files
df = dd.read_csv("data/*.csv")
```

### Converting from Pandas

If you're working with pandas DataFrames, convert them with:

```python
import pandas as pd
import dask.dataframe as dd

# Your pandas DataFrame
pandas_df = pd.read_csv("data.csv")

# Convert to Dask DataFrame
dask_df = dd.from_pandas(pandas_df, npartitions=4)  # adjust partitions based on your data size
```

**Tip:** The number of partitions should be chosen based on your dataset size and available memory. A good rule of thumb is to aim for partitions of 100-500 MB each.

## Ibis

Datatune integrates with Ibis to operate on a variety of backends including DuckDB, PostgreSQL, DataFusion, and more. This allows you to:

- Easily switch between different database backends
- Leverage the power of SQL databases for data transformations
- Work with data without loading it entirely into memory

### DuckDB Backend

```python
import ibis

# Connect to DuckDB
con = ibis.duckdb.connect("test.duckdb")

# Read CSV into DuckDB
people_view = con.read_csv(
    "data/organizations.csv",
    table_name="organizations"
)

# Create a table
con.create_table(
    "organizations",
    people_view,
    overwrite=True
)

# Get table reference
table = con.table("organizations")
```

### Using Datatune with Ibis Tables

Once you have an Ibis table, you can use it directly with Datatune operations:

```python
import datatune as dt
from datatune.llm.llm import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo")

# Transform Ibis tables using Datatune
mapped = dt.map(
    prompt="Extract Sub-Category from industry column.",
    output_fields=["Sub-Category"],
    input_fields=["Industry"]
)(llm, table)  # pass your Ibis table

# Filter the mapped Ibis table expression
filtered = dt.filter(
    prompt="Keep only countries in Asia.",
    input_fields=["Country"]
)(llm, mapped)

# Execute to get pandas DataFrame
result = filtered.execute()
```

### Other Ibis Backends

Ibis supports many backends. Here are some examples:

#### PostgreSQL
```python
con = ibis.postgres.connect(
    host="localhost",
    database="mydb",
    user="user",
    password="password"
)
table = con.table("my_table")
```

#### SQLite
```python
con = ibis.sqlite.connect("database.db")
table = con.table("my_table")
```

#### BigQuery
```python
con = ibis.bigquery.connect(
    project_id="my-project",
    dataset_id="my_dataset"
)
table = con.table("my_table")
```

## Choosing the Right Backend

| Backend | Best For | Key Benefits |
|---------|----------|--------------|
| **Dask** | Large CSV/Parquet files, distributed computing | Parallel processing, handles data larger than memory |
| **Ibis + DuckDB** | Fast analytical queries, embedded analytics | SQL interface, fast aggregations, no server needed |
| **Ibis + PostgreSQL** | Production databases, transactional data | ACID compliance, existing database infrastructure |
| **Ibis + BigQuery** | Cloud data warehouses, very large datasets | Serverless, scales automatically, integrates with GCP |

## Working with Transformed Data

After applying Datatune transformations, finalize and save your results:

### With Dask
```python
# Finalize removes metadata columns
result = dt.finalize(transformed_df)

# Save to CSV
result.compute().to_csv("output.csv", index=False)

# Save to Parquet
result.to_parquet("output.parquet")
```

### With Ibis
```python
# Execute returns a pandas DataFrame
result = transformed_table.execute()

# Save to CSV
result.to_csv("output.csv", index=False)

# Or write back to database
con.create_table("results_table", result, overwrite=True)
```

## Performance Tips

1. **Partition wisely**: For Dask, choose partition sizes that balance parallelism with overhead (100-500 MB per partition is a good starting point)

2. **Select relevant columns**: Use `input_fields` parameter to send only necessary columns to the LLM, reducing token usage and cost

3. **Set rate limits**: Configure `tpm` (tokens per minute) and `rpm` (requests per minute) to match your API limits and avoid throttling

4. **Use appropriate backends**: DuckDB for fast analytical queries, Dask for very large files, PostgreSQL for production data