# Datatune

Your backend for LLM powered Big Data Apps

## Installation
### From pip
```python
pip install datatune
```

### From source
```python
git clone https://www.github.com/farizrahman4u/datatune.git
cd datatune
pip install e .
```


## Getting Started

### Data Sources

Import datasets from a variety of sources:

```
import datatune as dt

ds1 = dt.dataset("hf://....")
ds2 = dt.dataset("s3://...")
```

Basic operations on datasets to yield views:
```

ds1 = ds1.filter("<filter condition>")
ds2 = ds2.transform(transform_fn)
ds3 = dt.concat([ds1, ds2])

```



### Apps
```python
from datatune.apps import TableQA

table_qa = TableQA(ds3)
table_qa.cli()
```

```python
from datatune.apps import LLMTransform

llm_transform = LLMTransform("if column A value is greater than 3, return red, else green.")

ds4 = llm_transform(ds3)  # or ds4 = ds3.transform(llm_transform)?

# TODO, similarly LLMFilter
```

### Stream data for training

```python
dataloader = ds3.pytorch()
```
