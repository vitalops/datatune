# 🎵 Datatune

[![PyPI version](https://img.shields.io/pypi/v/datatune.svg)](https://pypi.org/project/datatune/)
[![Python Versions](https://img.shields.io/pypi/pyversions/datatune.svg)](https://pypi.org/project/datatune/)
[![License](https://img.shields.io/github/license/vitalops/datatune)](https://github.com/vitalops/datatune/blob/main/LICENSE)

Perform transformations on your data with Natural language using LLMs

## Installation

```bash
pip install datatune
```

From source:

```
pip install -e .
```
## Quick Start

```
from datatune.core.map import Map
from datatune.core.filter Filter
from your_llm import llm  # Your LLM function
import dask.dataframe as dd
from datatune.core.op import finalize

# Load data from your source with Dask
df = dd.read_csv("products.csv")

# Transform data with Map
mapped = Map(
    prompt="Create a short title and extract categories from the description",
    output_fields=["title", "category", "subcategory"]
)(llm, df)

# Filter data based on criteria
filtered = Filter(
    prompt="Keep only products suitable for beginners"
)(llm, mapped)

# Get the final dataframe after cleanup of metadata and deleted rows after operations using `finalize`.
result = finalize(filtered)
result.compute().to_csv("beginner_products.csv")
```

## Features

### Map Operation

Transform data with natural language:

```
customers = dd.read_csv("customers.csv")
mapped = Map(
    prompt="Extract country and city from the address field",
    output_fields=["country", "city"]
)(llm, customers)
```

### Filter operation

```
# Filter to marketable products only
marketable = Filter(
    prompt="Keep only customers who are from Asia"
)(llm, mapped)
```

### Multiple LLM Support
Datatune works with various LLM providers:

```
# Using Ollama
from datatune.llm.llm import Ollama
llm = Ollama()

# Using Azure OpenAI
from datatune.llm.llm import Azure
llm = Azure(
    model_name="gpt-35-turbo",
    api_key=api_key,
    api_base=api_base,
    api_version=api_version)
```

More examples in the examples/ folder.

## License
MIT License
