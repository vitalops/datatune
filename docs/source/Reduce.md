# Reduce

The `Reduce` operation in Datatune reduces the size of a dataset by grouping, deduplicating, or otherwise collapsing rows based on a specified action.  
It is typically used to minimize downstream processing cost (e.g. LLM calls) by identifying canonical rows and eliminating redundant ones.


## Basic Usage

```python
import datatune as dt
from datatune.llm.llm import LLM
import dask.dataframe as dd

# Initialize LLM
llm = LLM(model_name="openai/gpt-3.5-turbo")

# Load data
df = dd.read_csv("data.csv")

# Get deduplication clusters
clusters = dt.reduce(df, action="dedup", embedding_model="text-embedding-3-small", llm=llm)


# Apply transformation
mapped_df = dt.map(
    prompt="Extract country and city from the address field",
    output_fields=["country", "city"]
    clusters=clusters             # pass clusters
)(llm, df)

# Process results
mapped_df.compute()
```
## Parameters
**df** (DataFrame, required): Input dataset to be reduced.

**action** (str, required): Name of the reduction action to apply. Examples: "dedup"

**kwargs** (optional): Configuration parameters forwarded to the selected reduction action.The accepted parameters depend on the specific action being used.

## Actions

Reduction behavior is defined by actions, which are registered internally and selected by name at runtime.
Reduce accepts generic parameters, plus action-specific parameters
that are interpreted only by the selected action.

Each action:

- Is implemented as a callable class

- Receives configuration via its constructor

Operates on the input DataFrame when called


### dedup

- Identifies semantically duplicate rows and selects a canonical representative for each group. Outputs a "dedup map"
```python
[{'canonical_id': 5, 'duplicate_ids': [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}]
```
This map can be passed Datatune operations which will only send the canonical rows to LLM API for transformation and transmit it's results to the respective duplicate rows thereby reducing tokens and therefore cost.

The dedup action works by first embedding all input rows into vector representations, which are streamed to disk to avoid loading the full dataset into memory. These embeddings are then indexed using FAISS and searched to identify clusters of semantically similar rows based on the configured similarity threshold. Finally, within each candidate cluster, an LLM evaluation step is applied to confirm true duplicates and select a canonical representative for each cluster.

#### Dedup Parameters

When `action="dedup"`, the following additional parameters may be passed
to `reduce`:

**llm** (Callable, required)

LLM callable used for semantic evaluation and final duplicate confirmation.

---

**embedding_model** (str, optional)

Embedding model used to convert rows into vector representations.

Default: `"text-embedding-3-small"`

---

**sim_threshold** (float, optional)

Cosine similarity threshold above which two rows are considered potential
duplicates.

Higher values result in stricter deduplication.

Default: `0.90`

---

**top_k** (int, optional)

Number of nearest neighbors retrieved per row during similarity search.

Default: `50`

---

**hnsw_m** (int, optional)

HNSW graph connectivity parameter controlling the number of bi-directional
links created for each node.

Higher values improve recall but increase memory usage.

Default: `32`

---

**ef_search** (int, optional)

Size of the dynamic candidate list during HNSW search.

Higher values improve search accuracy at the cost of latency.

Default: `64`

---

