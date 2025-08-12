# 🎵 Datatune

[![PyPI version](https://img.shields.io/pypi/v/datatune.svg)](https://pypi.org/project/datatune/)
[![License](https://img.shields.io/github/license/vitalops/datatune)](https://github.com/vitalops/datatune/blob/main/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/datatune)](https://pepy.tech/projects/datatune)
[![Docs](https://img.shields.io/badge/docs-docs.datatune.ai-blue)](https://docs.datatune.ai)

Transform your data with natural language using LLMs

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
df = dd.read_csv("customer_feedback.csv")
print(df.head())

# Extract insights from customer feedback
mapped = dt.Map(
    prompt="Extract customer sentiment and main concern from feedback text",
    output_fields=["sentiment", "main_concern"],
    input_fields = ["feedback_text"] # Relevant input fields (optional)
)(llm, df)

# Filter for negative feedback requiring immediate attention
urgent = dt.Filter(
    prompt="Keep only negative feedback that indicates urgent issues",
    input_fields = ["sentiment", "main_concern"]
)(llm, mapped)

# Get the final dataframe
result = dt.finalize(urgent)
result.compute().to_csv("urgent_feedback.csv")

print(result.head())
```

**customer_feedback.csv**
```
   CustomerID                                      feedback_text    date
0        1001  "The product quality is terrible and arrived damaged"  2024-01-15
1        1002  "Great service, very happy with my purchase"           2024-01-16  
2        1003  "Website is broken, can't complete my order"           2024-01-17
3        1004  "Love this product, will definitely buy again"         2024-01-18
4        1005  "Billing error on my account, need immediate help"     2024-01-19
```

**urgent_feedback.csv**
```
   CustomerID                                      feedback_text         sentiment       main_concern
0        1001  "The product quality is terrible and arrived damaged"  Negative    Product Quality
1        1003  "Website is broken, can't complete my order"           Negative    Technical Issue  
2        1005  "Billing error on my account, need immediate help"     Negative    Billing Error
```

## Features

### Map Operation

Transform data with natural language:

```python
# Classify support tickets by priority
tickets = dd.read_csv("support_tickets.csv")
classified = dt.Map(
    prompt="Classify by priority (Low/Medium/High) and extract estimated resolution time",
    output_fields=["priority", "estimated_hours"],
    input_fields=["subject", "description"]
)(llm, tickets)
```

**Output:**
```
   TicketID                    Subject         Priority  Estimated_Hours
0      2001    "Website login broken"          High              2
1      2002    "Update billing info"           Low               1  
2      2003    "Payment processing error"      Critical          4
```

### Example 1: Data Anonymization

Protect sensitive information while preserving data utility:

```python
# Anonymize personally identifiable information
customer_data = dd.read_csv("customer_records.csv")
anonymized = dt.Map(
    prompt="Replace all personally identifiable fields with XX",
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

Extract and categorize information:

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

## Multiple LLM Support

Datatune works with various LLM providers with the help of LiteLLM under the hood:

```python
# Using Ollama for local processing
from datatune.llm.llm import Ollama
llm = Ollama()

# Using Azure OpenAI
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

## Agents

Datatune provides an agentic framework which allows you to deploy agents that can generate and execute python scripts with datatune operations:

```python
import datatune as dt
from datatune.llm.llm import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo", tpm=200000)
agent = dt.Agent(llm)

# Let the agent decide the best transformation approach
df = agent.do("Clean this data and extract key business insights for reporting", df)
```

- This allows for intelligent operation selection based on the given prompt

## Data Compatibility

Datatune uses Dask DataFrames for scalable processing. Convert pandas DataFrames with:

```python
import dask.dataframe as dd
dask_df = dd.from_pandas(pandas_df, npartitions=4)
```

## Performance Tips

Reduce costs by specifying relevant input fields:

```python
# Only send relevant columns to the LLM
result = dt.Map(
    prompt="Extract sentiment from reviews",
    output_fields=["sentiment"],
    input_fields=["review_text"]  # Reduces token usage
)(llm, df)
```

If you don't set `rpm` or `tpm`, Datatune automatically uses optimal limits for your model.

## Resources

- [Examples](https://github.com/vitalops/datatune/tree/main/examples)
- [Documentation](https://docs.datatune.ai/)
- [Issues](https://github.com/vitalops/datatune/issues)
- Email: hello@vitalops.ai

## License
MIT License
