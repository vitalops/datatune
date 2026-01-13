# 🎵 Datatune

[![PyPI version](https://img.shields.io/pypi/v/datatune.svg)](https://pypi.org/project/datatune/)
[![License](https://img.shields.io/github/license/vitalops/datatune)](https://github.com/vitalops/datatune/blob/main/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/datatune)](https://pepy.tech/projects/datatune)
[![Docs](https://img.shields.io/badge/docs-docs.datatune.ai-blue)](https://docs.datatune.ai)
[![Discord](https://img.shields.io/badge/Discord-7289da?logo=discord&logoColor=white)](https://discord.gg/3RKA5AryQX)

Scalable Data Transformations with row-level intelligence.


## How It Works

![How it works](https://raw.githubusercontent.com/vitalops/datatune/main/how%20it%20works.png)

## Installation

```bash
pip install datatune
```

## Quick Start

```python
import datatune as dt
from datatune.llm.llm import OpenAI
import dask.dataframe as dd

llm = OpenAI(model_name="gpt-3.5-turbo")
df = dd.read_csv("products.csv")

# Extract categories using natural language
mapped = dt.map(
    prompt="Extract categories from the description and name of product.",
    output_fields=["Category", "Subcategory"],
    input_fields=["Description", "Name"]
)(llm, df)

# Filter with simple criteria
filtered = dt.filter(
    prompt="Keep only electronics products",
    input_fields=["Name"]
)(llm, mapped)

# Save results
result = dt.finalize(filtered)
result.compute().to_csv("electronics_products.csv")
```

## 🤖 Agents - Even Simpler

Let AI automatically figure out the transformation steps for you:

```python
import datatune as dt
from datatune.llm.llm import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo")
agent = dt.Agent(llm)

# Just describe what you want - the agent handles map, filter, and more
df = agent.do("Add ProfitMargin column and keep only African organizations", df)
result = dt.finalize(df)
```

The agent automatically:
- Determines which operations to use (map, filter, etc.)
- Chains multiple transformations
- Handles complex multi-step tasks from a single prompt

## Supported LLMs

```python
# OpenAI
from datatune.llm.llm import OpenAI
llm = OpenAI(model_name="gpt-3.5-turbo")

# Ollama (local)
from datatune.llm.llm import Ollama
llm = Ollama()

# Azure
from datatune.llm.llm import Azure
llm = Azure(model_name="gpt-3.5-turbo", api_key=api_key)
```

## Data Sources

Works with **Dask** and **Ibis** (DuckDB, PostgreSQL, BigQuery, and more):

```python
# Dask
import dask.dataframe as dd
df = dd.read_csv("data.csv")

# Ibis + DuckDB
import ibis
con = ibis.duckdb.connect("data.duckdb")
table = con.table("my_table")
```

## Learn More

- **[Documentation](https://docs.datatune.ai/)** - Complete guides and API reference
- **[Examples](https://github.com/vitalops/datatune/tree/main/examples)** - Real-world use cases
- **[Discord](https://discord.gg/3RKA5AryQX)** - Community support
- **[Issues](https://github.com/vitalops/datatune/issues)** - Report bugs or request features

## License

MIT License
