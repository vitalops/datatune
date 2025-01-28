
# view


## view.View


**Args:**

* **id (str)** - The unique identifier of the view
* **workspace (Workspace)** - The workspace object that contains this view

## view.dataset_slices


**Returns:**

List[DatasetSlice]: List of DatasetSlice objects representing portions of datasets in the view

## view.extra_columns


**Returns:**

List[ExtraColumn]: List of ExtraColumn objects associated with the view

## view.extend


**Args:**

* **data (Union[Dataset, DatasetSlice])** - The dataset or slice to add to the view
* **table_index (Optional[int])** - Index of the table in the dataset. Defaults to 0
* **start (Optional[int])** - Starting index for the slice
* **stop (Optional[int])** - Ending index for the slice
* **column_maps (Optional[Dict[str, str]])** - Mapping of source column names to target column names

**Returns:**

View: The updated view object

**Raises:**

TypeError: If data is neither a Dataset nor a DatasetSlice

## view.add_extra_column


**Args:**

* **column_name (str)** - Name of the new column
* **column_type (str)** - Data type of the column
* **labels (Optional[List[str]])** - List of possible labels for categorical columns
* **default_value (Any)** - Default value for the column
* **num_rows (int)** - Number of rows to initialize. Defaults to 0

**Returns:**

View: The updated view object

## view.get_extra_column


**Args:**

* **id (str)** - ID of the extra column to retrieve

**Returns:**

ExtraColumn: The requested extra column object

## view.delete_extra_column


**Args:**

* **id (str)** - ID of the extra column to delete