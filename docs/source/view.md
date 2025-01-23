# view

## view.View

A class representing a view in Datatune that can combine and manipulate datasets.

Args:
    id (str): The unique identifier of the view
    workspace (Workspace): The workspace object that contains this view

## view.dataset_slices

Returns a list of dataset slices associated with this view.

Returns:
    List[DatasetSlice]: List of DatasetSlice objects representing portions of datasets in the view

## view.extra_columns

Returns a list of extra columns added to this view.

Returns:
    List[ExtraColumn]: List of ExtraColumn objects associated with the view

## view.extend

Extends the view with data from a dataset or dataset slice.

Args:
    data (Union[Dataset, DatasetSlice]): The dataset or slice to add to the view
    table_index (Optional[int]): Index of the table in the dataset. Defaults to 0
    start (Optional[int]): Starting index for the slice
    stop (Optional[int]): Ending index for the slice
    column_maps (Optional[Dict[str, str]]): Mapping of source column names to target column names

Returns:
    View: The updated view object

Raises:
    TypeError: If data is neither a Dataset nor a DatasetSlice

## view.add_extra_column

Adds a new column to the view.

Args:
    column_name (str): Name of the new column
    column_type (str): Data type of the column
    labels (Optional[List[str]]): List of possible labels for categorical columns
    default_value (Any): Default value for the column
    num_rows (int): Number of rows to initialize. Defaults to 0

Returns:
    View: The updated view object

## view.get_extra_column

Retrieves an extra column by its ID.

Args:
    id (str): ID of the extra column to retrieve

Returns:
    ExtraColumn: The requested extra column object

## view.delete_extra_column

Deletes an extra column from the view.

Args:
    id (str): ID of the extra column to delete

