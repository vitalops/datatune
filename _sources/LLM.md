# LLM

The `LLM` module in Datatune provides classes for interfacing with various Language Model providers. It handles the connection to and inference from LLMs with a unified API structure, supporting both single and batch inference, with the help of LiteLLM.

## Base LLM Class

The base `LLM` class provides the foundation for working with different LLM providers.

```python
from datatune.llm.llm import LLM

# Initialize with OpenAI model
llm = LLM(model_name="openai/gpt-3.5-turbo")

# Process a single prompt
response = llm("Extract the city name from: '123 Main St, New York, NY 10001'")

# Process multiple prompts in batch
responses = llm(["Extract city from: 'NY, USA'", "Extract city from: 'Paris, France'"])
```

### Parameters

- **model_name** (str, required): The identifier for the LLM model to use.

- **temperature** (float, optional): Controls randomness in output generation. Defaults to 0.0 for deterministic results.

- **kwargs**: Additional parameters to pass to the underlying LLM provider.

## Specialized LLM Providers

### Ollama

The `Ollama` class provides an interface to locally hosted models via Ollama.

```python
from datatune.llm.llm import Ollama

# Initialize with default parameters (gemma3:4b on localhost)
llm = Ollama()

# Or specify a different model and endpoint
llm = Ollama(
    model_name="llama3:15b", 
    api_base="http://ollama-server:11434"
)
```

#### Parameters

- **model_name** (str, optional): The model to use. Defaults to "gemma3:4b".

- **api_base** (str, optional): The base URL for the Ollama API. Defaults to "http://localhost:11434".

- **kwargs**: Additional parameters to pass to the Ollama API.

### Azure

The `Azure` class provides an interface to Azure OpenAI models.

```python
from datatune.llm.llm import Azure
import os

# Initialize with Azure configuration
llm = Azure(
    model_name="gpt-35-turbo",
    api_key=os.getenv("AZURE_API_KEY"),
    api_base=os.getenv("AZURE_API_BASE"),
    api_version=os.getenv("AZURE_API_VERSION")
)
```

#### Parameters

- **model_name** (str, required): The name of the deployed model in Azure.

- **api_key** (str, optional): The API key for Azure OpenAI. If None, will try to use environment variables.

- **api_base** (str, optional): The base URL for the Azure OpenAI API.

- **api_version** (str, optional): The API version to use.

- **kwargs**: Additional parameters to pass to the Azure API.

## Usage in Datatune Operations

LLM instances are passed to Datatune operations like `Map` and `Filter`:

```python
from datatune.core.map import Map
from datatune.llm.llm import LLM
import dask.dataframe as dd

# Initialize LLM
llm = LLM(model_name="openai/gpt-3.5-turbo")

# Load data
df = dd.read_csv("data.csv")

# Use LLM with a Map operation
mapped_df = Map(
    prompt="Extract company name from the text",
    output_fields=["company_name"]
)(llm, df)
```