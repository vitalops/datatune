# 🎵 Datatune

[![PyPI version](https://img.shields.io/pypi/v/datatune.svg)](https://pypi.org/project/datatune/)
[![License](https://img.shields.io/github/license/vitalops/datatune)](https://github.com/vitalops/datatune/blob/main/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/datatune)](https://pepy.tech/projects/datatune)
[![Docs](https://img.shields.io/badge/docs-docs.datatune.ai-blue)](https://docs.datatune.ai)

Perform transformations on your data with natural language using LLMs

## Installation

```bash
pip install datatune
```

From source:

```bash
pip install -e .
```

## Quick Start
```python
import os
import dask.dataframe as dd

import datatune as dt
from datatune.llm.llm import OpenAI

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Set tokens-per-minute and requests-per-minute limits 
llm = OpenAI(model_name="gpt-3.5-turbo", tpm = 200000, rpm = 50)

# Load data from your source with Dask
df = dd.read_csv("tests/test_data/products.csv")
print(df.head())

# Transform data with Map
mapped = dt.Map(
    prompt="Extract categories from the description and name of product.",
    output_fields=["Category", "Subcategory"],
    input_fields = ["Description","Name"] # Relevant input fields (optional)
)(llm, df)

# Filter data based on criteria
filtered = dt.Filter(
    prompt="Keep only electronics products",
    input_fields = ["Name"] # Relevant input fields (optional)
)(llm, mapped)

# Get the final dataframe after cleanup of metadata and deleted rows after operations using `finalize`.
result = dt.finalize(filtered)
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

If you don't set `rpm` or `tpm`, Datatune will automatically look up default limits for your model from our [model_rate_limits](datatune/llm/model_rate_limits.py). If model is not available in the lookup dictionary rpm and tpm will default to **gpt-3.5-turbo** limits.

Passing `input_fields` reduces the number of tokens sent by sending only relevant columns as input to the given LLM API, hence reducing the cost.

## Features

### Example 1: Data Anonymization

Protect sensitive information while preserving data utility:

```python
# Anonymize personally identifiable information
customer_data = dd.read_csv("customer_records.csv")
anonymized = dt.Map(
    prompt="Replace all personally identifiable fields with XX - emails, phone numbers, names, addresses",
    output_fields=["anonymized_text"],
    input_fields=["customer_notes"]
)(llm, customer_data)
```

**Output:**
```
   CustomerID                           Original_Notes                    Anonymized_Text
0        3001    "John Smith called about bill"         "XX called about bill"
1        3002    "Email: jane@email.com for updates"   "Email: XX for updates"
2        3003    "Call 555-1234 regarding order"       "Call XX regarding order"
```

### Example 2: Data Classification

Extract and categorize information from unstructured data:

```python
# Classify customer support emails by department and urgency
support_emails = dd.read_csv("support_emails.csv")
classified = dt.Map(
    prompt="Classify emails by department (Technical/Billing/Sales) and urgency level (Low/Medium/High/Critical)",
    output_fields=["department", "urgency_level", "estimated_response_time"],
    input_fields=["subject", "email_body"]
)(llm, support_emails)
```

**Output:**
```
   EmailID                    Subject         Department  Urgency_Level  Estimated_Response_Time
0     4001    "Login issues on mobile"      Technical        High              "2 hours"
1     4002    "Invoice payment question"   Billing          Medium            "1 day"  
2     4003    "Server completely down"     Technical        Critical          "30 minutes"
```

### Example 3: Smart Filtering

Filter to remove rows based on criteria:

```python
# Filter high-quality product reviews
reviews = dd.read_csv("reviews.csv")
quality_reviews = dt.Filter(
    prompt="Keep only genuine, detailed reviews that are not spam",
    input_fields=["review_text", "reviewer_history"]
)(llm, reviews)
```

**Output:**
```
   ReviewID                           Review_Text              Reviewer_History    Rating
0      5001    "Excellent product, works as expected..."    "50+ reviews, verified"   5
1      5004    "Good value for money, fast shipping..."     "25+ reviews, verified"   4  
2      5007    "Quality exceeded my expectations..."        "15+ reviews, verified"   5
```

### Map Operation

Transform data with natural language:

```python
customers = dd.read_csv("customers.csv")
mapped = dt.Map(
    prompt="Extract country and city from the address field",
    output_fields=["country", "city"]
)(llm, customers)
```

### Filter operation

```python
# Filter to remove rows
filtered = dt.Filter(
    prompt="Keep only customers who are from Asia"
)(llm, mapped)
```

### Multiple LLM Support

Datatune works with various LLM providers with the help of LiteLLM under the hood:

```python
# Using Ollama
from datatune.llm.llm import Ollama
llm = Ollama()

# Using Azure
from datatune.llm.llm import Azure
llm = Azure(
    model_name="gpt-3.5-turbo",
    api_key=api_key,
    api_base=api_base,
    api_version=api_version)

# OpenAI
from datatune.llm.llm import OpenAI
llm = OpenAI(model_name="gpt-3.5-turbo")
```

### Agents

Datatune provides an agentic framework which allows you to deploy agents that can generate and execute python scripts with datatune operations:

```python
import datatune as dt
from datatune.llm.llm import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo", tpm=200000)

# Initialize an agent by providing an LLM
agent = dt.Agent(llm)
prompt = "your prompt for data transformation"

# Transform your dask DataFrame
df = agent.do(prompt, df)
```

- This allows for intelligent operation selection based on the given prompt

### Data Compatibility

Datatune leverages Dask DataFrames to enable scalable processing across large datasets. This approach allows you to:

- Process data larger than context length of LLMs
- Execute parallel computations efficiently

If you're working with pandas DataFrames, convert them with a simple:

```python
import dask.dataframe as dd
dask_df = dd.from_pandas(pandas_df, npartitions=4)  # adjust partitions based on your data size
```

### Examples
Check out [examples](https://github.com/vitalops/datatune/tree/main/examples)

### Documentation

Check out our [documentation](https://docs.datatune.ai/) to learn how to use datatune.

### Issues 

Want to raise an issue or want us to build a new feature?
Head over to [issues](https://github.com/vitalops/datatune/issues) and raise a ticket!   

You can also mail us at hello@vitalops.ai

## License
MIT License
