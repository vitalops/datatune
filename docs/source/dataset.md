# dataset

## dataset.Dataset

A class representing a dataset in Datatune.

Args:
    id (str): The unique identifier of the dataset
    workspace (Workspace): The workspace object that contains this dataset

Attributes:
    id (str): The dataset's unique identifier
    workspace (Workspace): The workspace containing this dataset
    tables (List): List of tables in the dataset

## dataset.DatasetSlice

A class representing a slice or subset of a dataset.

Args:
    dataset (Dataset): The dataset to slice
    table_index (Optional[int]): Index of the table in the dataset. Defaults to 0
    start (Optional[int]): Starting index of the slice. None means start from beginning
    stop (Optional[int]): Ending index of the slice. None means go until the end

Attributes:
    dataset (Dataset): The dataset being sliced
    table_index (int): Index of the table in the dataset
    start (Optional[int]): Starting index of the slice
    stop (Optional[int]): Ending index of the slice
    table_id (str): ID of the table being sliced

## dataset.entity

Returns the entity associated with this dataset's workspace.

## dataset.name

Returns the name of the dataset.

## dataset.api

Returns the API instance associated with this dataset's workspace.

