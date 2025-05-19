# 🎵 Datatune

[![PyPI version](https://img.shields.io/pypi/v/datatune.svg)](https://pypi.org/project/datatune/)
[![Python Versions](https://img.shields.io/pypi/pyversions/datatune.svg)](https://pypi.org/project/datatune/)
[![License](https://img.shields.io/github/license/vitalops/datatune)](https://github.com/vitalops/datatune/blob/main/LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/datatune.svg)](https://pypi.org/project/datatune/)

Perform transformations on your data with natural language using LLMs

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
import os
import dask.dataframe as dd

from datatune.core.map import Map
from datatune.core.filter import Filter
from datatune.llm.llm import LLM
from datatune.core.op import finalize

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
llm = LLM(model_name="gpt-35-turbo")

# Load data from your source with Dask
df = dd.read_csv("tests/test_data/products.csv")
print(df.head())

# Transform data with Map
mapped = Map(
    prompt="Extract categories from the description.",
    output_fields=["Category", "Subcategory"]
)(llm, df)

# Filter data based on criteria
filtered = Filter(
    prompt="Keep only electronics products"
)(llm, mapped)

# Get the final dataframe after cleanup of metadata and deleted rows after operations using `finalize`.
result = finalize(filtered)
result.compute().to_csv("electronics_products.csv")

new_df = dd.read_csv("electronics_products.csv")
print(new_df.head())
```

**products.csv**
```
   ProductID             Name   Price  Quantity                                        Description      SKU
0       1001   Wireless Mouse   25.99       150  Ergonomic wireless mouse with 2.4GHz connectivity  WM-1001
1       1002     Office Chair   89.99        75  Comfortable swivel office chair with lumbar su...  OC-2002
2       1003       Coffee Mug    9.49       300                  Ceramic mug, 12oz, microwave safe  CM-3003
3       1004  LED Monitor 24"  149.99        60  24-inch Full HD LED monitor with HDMI and VGA ...  LM-2404
4       1005    Notebook Pack    6.99       500          Pack of 3 ruled notebooks, 100 pages each  NP-5005
```

**electronics_products.csv**
```
   Unnamed: 0  ProductID               Name  ...      SKU     Category           Subcategory
0           0       1001     Wireless Mouse  ...  WM-1001  Electronics  Computer Accessories
1           3       1004    LED Monitor 24"  ...  LM-2404  Electronics              Monitors
2           6       1007     USB-C Cable 1m  ...  UC-7007  Electronics                Cables
3           8       1009  Bluetooth Speaker  ...  BS-9009  Electronics                 Audio
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

More examples in the [examples/](https://github.com/vitalops/datatune/tree/main/examples)folder.

## License
MIT License
