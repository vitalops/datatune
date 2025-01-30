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

## Blogs

:::::{grid} 2
::::{grid-item-card} Simplifying LLM Training with DataTune: A Beginner's Guide
:link: https://medium.com/@abhijithneilabraham/simplifying-llm-training-with-datatune-a-beginners-guide-4492c6ca5812
Learn how to get started with DataTune for LLM training workflows.
::::

::::{grid-item-card} Simplify Large Data Processing with DataTune
:link: https://medium.com/@abhijithneilabraham/simplify-large-data-processing-with-datatune-01ecfb736bcd
Explore techniques for handling and processing large datasets efficiently with DataTune.
::::
:::::

## SDK Examples
:::::{grid} 2
::::{grid-item-card} Jupyter Notebook Examples
:link: https://github.com/vitalops/datatune/blob/main/examples/sdk_example.ipynb
Explore practical examples of using the DataTune SDK.
::::

:::::

## Documentation

```{toctree}
:maxdepth: 2
:caption: Contents:

streaming.md
extra_column.md
workspace.md
entity.md
dataset.md
view.md
```
## License

MIT License