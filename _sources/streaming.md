
# streaming


## streaming.DataTuneLoader


**Args:**

* **view (View)** - The view object to stream data from
* **start_index (int)** - The starting index for streaming. Defaults to 0
* **end_index (Optional[int])** - The ending index for streaming. Defaults to None (stream until end)
* **batch_size (Optional[int])** - Number of samples per batch. Defaults to 32
* **columns (Optional[List[str]])** - List of column names to fetch. Defaults to None (fetch all columns)
* **num_workers (Optional[int])** - Number of worker threads. Defaults to 1
* **Attributes** - 
* **cache (List)** - Internal cache for storing pre-fetched batches
* **cache_size (int)** - Current size of cached data
* **thread (threading.Thread)** - Background thread for data fetching
* **_view_size (int)** - Total number of rows in the view

## streaming._get_view_size


**Returns:**

int: Total number of rows in the view

## streaming._get_batch


**Args:**

* **start_index (int)** - Starting index of the batch
* **end_index (int)** - Ending index of the batch

**Returns:**

Dict: Dictionary containing the batch data and number of samples

## streaming.bg_thread


## streaming.reset


## streaming.stream


**Raises:**

Exception: If there's an error fetching a batch
AssertionError: If a requested column is not found in the batch

## streaming.__iter__


## streaming.__len__


**Returns:**

int: Number of batches in the dataset

## streaming.shutdown
