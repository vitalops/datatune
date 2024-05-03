from .exceptions import DatatuneException

class View:
    """
    Represents a view within a workspace on the Datatune platform.
    
    A view is a user-defined subset of data organized from one or more datasets. This class
    provides functionalities to modify and manage the structure of a view, such as extending it
    with additional dataset slices or adding new columns.

    Attributes:
        workspace (Workspace): The workspace to which this view belongs.
        name (str): The name of the view.
    
    Methods:
        extend(dataset_name, slice_range): Extends the view by adding a slice from a specified dataset.
        add_columns(column_name, column_type, default_value): Adds a new column to the view with specified attributes.
    """
    def __init__(self, workspace, name):
        self.workspace = workspace
        self.name = name

    def extend(self, dataset_name, slice_range):
        """Extends a view with a slice of a dataset."""
        data = {'dataset_name': dataset_name, 'slice_range': slice_range}
        response = self.workspace.api.put(f"workspaces/{self.workspace.workspace_name}/views/{self.name}",
                                          json=data)
        if not response.get('success'):
            raise DatatuneException(f"Failed to extend view '{self.name}'.")

    def add_columns(self, column_name, column_type, default_value):
        """Adds extra columns to a view."""
        data = {'column_name': column_name, 'column_type': column_type, 'default_value': default_value}
        response = self.workspace.api.post(f"workspaces/{self.workspace.workspace_name}/views/{self.name}/columns",
                                           json=data)
        if not response.get('success'):
            raise DatatuneException(f"Failed to add column '{column_name}' to view '{self.name}'.")
