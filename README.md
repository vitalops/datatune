# DataTune Client

[![PyPI version](https://badge.fury.io/py/datatune-client.svg)](https://badge.fury.io/py/datatune-client)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-datatune.ai-blue)](https://docs.datatune.ai)
[![Downloads](https://static.pepy.tech/badge/datatune-client)](https://pepy.tech/project/datatune-client)

DataTune is a unified platform for AI data workflows that simplifies how you manage, process, and stream training data for machine learning models.

The client sdk allows you to connect with the platform via code and continue with your data workflows with ease.

[Website](https://datatune.ai) • [Documentation](https://docs.datatune.ai) • [Blog](https://medium.com/@abhijithneilabraham/simplifying-llm-training-with-datatune-a-beginners-guide-4492c6ca5812) • [Quick Start](#quick-start)

## Key Features

- **Unified Data Management**: Centralize your datasets from multiple sources in one place
- **Efficient Streaming**: Stream data in batches directly to your machines
- **Simple Integration**: Get started with just a few lines of code
- **Intuitive Dataset Views**: Easy-to-use interface for data labeling and editing.
- **Parallel processing Support**: Scale your data processing with parallel workers

## Installation

```bash
pip install datatune-client
```

## Quick Start

```python
from datatune.api import API
from datatune.entity import Entity
from datatune.workspace import Workspace
from datatune.streaming import DataTuneLoader

# Initialize
api = API(api_key="your-api-key")
entity = Entity(id="your-org-id", api=api)
workspace = Workspace(entity=entity, name="your-workspace")

# Load view and start streaming
view = workspace.load_view("your-view-name")
dataloader = DataTuneLoader(view, batch_size=32, num_workers=4)

# Stream data
for batch in dataloader.stream():
    # Process your batch
    pass
```

## Documentation

For detailed documentation and guides, visit [docs.datatune.ai](https://docs.datatune.ai)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.