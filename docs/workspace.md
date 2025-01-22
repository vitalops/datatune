# workspace

## workspace.Workspace

A class representing a workspace in Datatune for managing datasets, views, and credentials.

Args:
    entity: The Entity object representing the organization
    id (Optional[str]): The workspace ID. If None, creates a new workspace
    name (Optional[str]): Name of the workspace. Defaults to "Awesome Workspace" if creating new
    description (Optional[str]): Description of the workspace. Defaults to "No description provided"

## workspace.delete

Deletes the current workspace.

## workspace.update

Updates the workspace's name and/or description.

Args:
    name (Optional[str]): New name for the workspace
    description (Optional[str]): New description for the workspace

## workspace.add_dataset

Adds a new dataset to the workspace.

Args:
    path (Union[str, List[str]]): Path or list of paths to the dataset files
    name (Optional[str]): Name of the dataset
    dataset_type (Optional[str]): Type of the dataset
    description (Optional[str]): Description of the dataset

Returns:
    Dataset: A Dataset object representing the newly added dataset

## workspace.load_dataset

Loads an existing dataset by ID.

Args:
    id (str): The ID of the dataset to load

Returns:
    Dataset: The loaded Dataset object

## workspace.delete_dataset

Deletes a dataset from the workspace.

Args:
    id (str): ID of the dataset to delete

## workspace.create_view

Creates a new view in the workspace.

Args:
    view_name (str): Name of the view to create

Returns:
    View: A View object representing the newly created view

## workspace.load_view

Loads an existing view by name.

Args:
    name (Union[str, View]): Name of the view or View object to load

Returns:
    View: The loaded View object

## workspace.delete_view

Deletes a view from the workspace.

Args:
    view_name (str): Name of the view to delete

## workspace.add_credentials

Adds new credentials to the workspace.

Args:
    name (str): Name of the credentials
    credential_type (str): Type of credentials
    credentials (Dict): Dictionary containing credential information
    path (Optional[str]): Path associated with the credentials
    description (Optional[str]): Description of the credentials

