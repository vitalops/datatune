from .exceptions import DatatuneException

class View:
    """
    Represents a view within a workspace on the Datatune platform.

    A view is a user-defined subset of data organized from one or more datasets. This class
    provides functionalities to modify and manage the structure of a view, such as extending it
    with additional dataset slices or adding new columns, as well as filtering and sorting the data.
    
    Attributes:
        workspace (Workspace): The workspace to which this view belongs.
        name (str): The name of the view.
    
    Methods:
        extend(dataset_name, slice_range): Extends the view by adding a slice from a specified dataset.
        add_columns(column_name, column_type, default_value): Adds a new column to the view with specified attributes.
        add_filter(column_name, condition): Adds a filter to the view.
        sort_by(column_name, order): Sorts the view based on a specified column.
        group_by(column_name): Groups the data within the view by a specified column.
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

    def add_filter(self, column_name, condition):
        """Adds a filter to the view based on a column and condition."""
        data = {'column_name': column_name, 'condition': condition}
        response = self.workspace.api.post(f"workspaces/{self.workspace.workspace_name}/views/{self.name}/filter",
                                           json=data)
        if not response.get('success'):
            raise DatatuneException("Failed to add filter to view.")

    def sort_by(self, column_name, order):
        """Sorts the view based on a specified column and order."""
        data = {'column_name': column_name, 'order': order}
        response = self.workspace.api.post(f"workspaces/{self.workspace.workspace_name}/views/{self.name}/sort",
                                           json=data)
        if not response.get('success'):
            raise DatatuneException("Failed to sort view.")

    def group_by(self, column_name):
        """Groups the data within the view by a specified column."""
        data = {'column_name': column_name}
        response = self.workspace.api.post(f"workspaces/{self.workspace.workspace_name}/views/{self.name}/group",
                                           json=data)
        if not response.get('success'):
            raise DatatuneException("Failed to group data in view.")
