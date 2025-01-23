# DataTune Client

[![PyPI version](https://badge.fury.io/py/datatune-client.svg)](https://badge.fury.io/py/datatune-client)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-datatune.ai-blue)](https://docs.datatune.ai)
[![Downloads](https://static.pepy.tech/badge/datatune-client)](https://pepy.tech/project/datatune-client)

SDK for [Datatune](https://datatune.ai), a unified platform for AI data workflows

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

## License

MIT License