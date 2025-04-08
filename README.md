# 🎵 Datatune

[![PyPI version](https://img.shields.io/pypi/v/datatune.svg)](https://pypi.org/project/datatune/)
[![Python Versions](https://img.shields.io/pypi/pyversions/datatune.svg)](https://pypi.org/project/datatune/)
[![License](https://img.shields.io/github/license/vitalops/datatune)](https://github.com/vitalops/datatune/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://vitalops.github.io/datatune/)
[![Downloads](https://static.pepy.tech/badge/datatune)](https://pepy.tech/project/datatune)

**Datatune** supercharges your data workflows with LLMs, enabling natural language operations on both structured and unstructured data.

```python
import datatune as dt

# Load your dataset
data = dt.load_dataset("customer_feedback.csv")
# or from an s3 bucket:
data = dt.load_dataset("s3://my_bucket/customer_feedback_parquets")

# Apply an LLM-powered transformation with natural language
data = data.map("extract sentiment and key topics from each review")

# Filter with semantic understanding
important_issues = data.filter("keep only reviews mentioning product defects or safety concerns")
```

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Key Features](#-key-features)
- [Examples](#-examples)
- [API Reference](#-api-reference)
- [Distributed Processing](#-distributed-processing)
- [LLM Configuration](#-llm-configuration)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Installation

### Using pip

```bash
pip install datatune
```

### From source

```bash
git clone https://github.com/vitalops/datatune.git
cd datatune
pip install -e .
```

### Prerequisites

Datatune requires:
- Python 3.8+
- An API key for your preferred LLM provider (OpenAI, Anthropic, etc.)

## 🏁 Quick Start

```python
import datatune as dt
import os

# Load a dataset
ds = dt.load_dataset("data.csv")

# Map operation using natural language
transformed_ds = ds.map("standardize dates in the 'event_date' column to YYYY-MM-DD format")

# Filter operation using natural language
filtered_ds = transformed_ds.filter("remove rows with missing critical information")

# Save the result
filtered_ds.to_csv("processed_data.csv")
```

## ✨ Key Features

### Natural Language Data Operations

- **Map**: Transform data using natural language instructions
- **Filter**: Select rows based on semantic criteria
- **Reduce**: Aggregate and summarize data with intelligence
- **Expand**: Generate synthetic data with specific characteristics

### Supported Data Sources

- CSV, JSON, Parquet files
- Pandas DataFrames
- SQL databases (via SQLAlchemy)
- Hugging Face datasets
- Apache Arrow tables
- In-memory data structures

### Processing Capabilities

- Text normalization and standardization
- Entity extraction and anonymization
- Sentiment and intent analysis
- Classification and categorization
- Data enrichment and augmentation
- Complex pattern recognition

## 📊 Examples

### Data Anonymization

```python
# Anonymize personal information
ds = dt.load_dataset("customer_data.csv")
anonymized = ds.map("replace all personally identifiable information with XXX")
```


### Data Generation

```python
# Create an empty dataset with schema
schema = {
    "transaction_id": "string",
    "amount": "float",
    "merchant": "string", 
    "category": "string",
    "description": "string",
    "timestamp": "datetime",
    "customer_id": "string"
}

# Generate 1000 synthetic transactions
empty_ds = dt.create_empty_dataset(schema)
synthetic_data = empty_ds.expand(1000, """
Generate realistic banking transactions with:
- Transaction amounts following typical consumer spending patterns
- Include occasional suspicious patterns (5% of transactions)
- Timestamps should follow realistic temporal patterns
- Ensure category and description align logically
""")
```

### Semantic Data Filtering

```python
# Patient eligibility filtering for clinical trials
patients = dt.load_dataset("patient_records.csv")

criteria = """
Include patients who:
- Have Type 2 diabetes
- HbA1c between 7.5% and 9.0%
- No history of cardiovascular events
- Failed on or intolerant to metformin
- Age 40-65
"""

eligible = patients.filter(f"Determine if patient meets these criteria: {criteria}")
```


### Data Transformation

```python
# Enrich product descriptions
products = dt.load_dataset("product_catalog.csv")
enriched = products.map("""
Enhance the product description by:
1. Adding 2-3 key feature highlights
2. Mentioning the ideal use case
3. Improving readability while maintaining accuracy
4. Ensuring a consistent professional tone
""")
```

### Development Setup

```bash
git clone https://github.com/vitalops/datatune.git
cd datatune
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest
```

## 🔬 Research

If you use Datatune in your research, please cite:

```bibtex
@software{datatune2024,
  author = {VitalOps Team},
  title = {Datatune: LLM-Powered Data Workflows},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/vitalops/datatune}
}
```

## 📜 License

Datatune is released under the [MIT License](https://github.com/vitalops/datatune/blob/main/LICENSE).
