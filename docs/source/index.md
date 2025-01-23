# DataTune Client

SDK for [Datatune](https://datatune.ai), a unified platform for AI data workflows.

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

## Resources

:::::{grid} 2

::::{grid-item-card} Latest Blog Posts
:link: https://medium.com/@abhijithneilabraham/simplifying-llm-training-with-datatune-a-beginners-guide-4492c6ca5812

**Simplifying LLM Training with DataTune: A Beginner's Guide**

Learn how to get started with DataTune for LLM training workflows.
::::

::::{grid-item-card} SDK Examples
:link: https://github.com/vitalops/datatune/blob/main/examples/sdk_example.ipynb

**Jupyter Notebook Examples**

Explore practical examples of using the DataTune SDK.
::::

:::::

## Documentation

```{toctree}
:maxdepth: 2
:caption: Contents:

entity.md
dataset.md
view.md
workspace.md
streaming.md
extra_column.md
build/html/_sources/entity.md
build/html/_sources/dataset.md
build/html/_sources/view.md
build/html/_sources/workspace.md
build/html/_sources/streaming.md
build/html/_sources/extra_column.md
```
## License

MIT License