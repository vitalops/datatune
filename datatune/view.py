import pandas as pd
from .exceptions import DatatuneException
from .query import Query


class View:
    """
    Represents a view within a workspace on the Datatune platform.

    A view is a user-defined subset of data organized from one or more datasets. This class
    provides functionalities to modify and manage the structure of a view, such as extending it
    with additional dataset slices or adding new columns, as well as filtering and sorting the data.
    
    Attributes:
        workspace (Workspace): The workspace to which this view belongs.
        name (str): The name of the view.

    """

    def __init__(self, workspace, name):
        self.workspace = workspace
        self.name = name
        self.query = Query(self)

    def extend(self, dataset_id, slice_range):
        """Extends a view with a slice of a dataset."""
        workspace_name = self.workspace.workspace_name
        view_name = self.name
        response = self.workspace.api.extend_view(workspace_name,
                                                  view_name,
                                                  dataset_id,
                                                  slice_range)
        if not response.get('success'):
            raise DatatuneException(f"Failed to extend view '{self.name}'.")
        return self


    def add_columns(self, data=None, column_name=None, column_type="string", default_value=None):
        """
        Adds columns to the view either from a DataFrame/Series or by specifying column details.

        Args:
            data (pd.DataFrame or pd.Series or str, optional): A DataFrame, Series, or path to a Parquet file containing the columns to add.
            column_name (str, optional): Name of the column to add if not adding from `data` and if `data` is a Series without a name.
            column_type (str, optional): Type of the column if `column_name` is specified.
            default_value (optional): Default value for the column if `column_name` is specified.

        Returns:
            View: The view instance to allow method chaining.

        Raises:
            DatatuneException: If the API call fails or the input is invalid.
        """
        if data is not None:
            if isinstance(data, pd.DataFrame):
                for column in data.columns:
                    self.workspace.api.add_column_to_view(
                        self.workspace.workspace_name,
                        self.name,
                        column,
                        data[column].dtype.name
                    )
            elif isinstance(data, pd.Series):
                column_name = data.name if data.name else column_name
                if not column_name:
                    raise ValueError("Column name must be provided for unnamed Series.")
                self.workspace.api.add_column_to_view(
                    self.workspace.workspace_name,
                    self.name,
                    column_name,
                    data.dtype.name
                )
            elif isinstance(data, str) and data.endswith('.parquet'):
                try:
                    df = pd.read_parquet(data)
                    for column in df.columns:
                        self.workspace.api.add_column_to_view(
                            self.workspace.workspace_name,
                            self.name,
                            column,
                            df[column].dtype.name
                        )
                except Exception as e:
                    raise DatatuneException(f"Failed to load data from {data}: {str(e)}")
            else:
                raise ValueError("Unsupported data input. Provide a DataFrame, Series, or a Parquet file path.")
        elif column_name:
            self.workspace.api.add_column_to_view(
                self.workspace.workspace_name,
                self.name,
                column_name,
                column_type,
                default_value
            )
        else:
            raise ValueError("Either provide a data source or column details.")
        return self

    def filter(self, condition):
        """
        Applies a filtering condition to the data within the view

        Args:
            condition (str): A SQL condition string used for filtering the data. 
                             This should be a valid SQL WHERE clause condition, such as "age > 30" or "status = 'active'".

        Returns:
            View: The current View instance with the applied filter condition.

        """
        response = self.query.filter(condition)
        if not response.get('success'):
            raise DatatuneException(f"Failed to filter view '{self.name}'.")
        return self
    
    def sort(self, columns, ascending=True):
        """
        Sorts the data in the view based on specified columns and order.

        Args:
            columns (str or list of str): The column or columns to sort the data by. If a list is provided,
                                          data will be sorted by multiple columns in the order specified in the list.
            ascending (bool, optional): Specifies the sort direction. True for ascending, False for descending.
                                        Default is True.

        Returns:
            View: The current View instance with the applied sorting.

        """
        response = self.query.sort(columns, ascending)
        if not response.get('success'):
            raise DatatuneException(f"Failed to sort view '{self.name}'.")
        return self

    def display(self, columns="*"):
        """
        Displays data from the view.

        Args:
            columns (str or list of str, optional): The column or columns to be selected for display.
                                                    Defaults to "*" which selects all columns.

        Returns:
            View: The current View instance after selecting the specified columns.

        """
        response = self.query.select(columns)
        if not response.get('success'):
            raise DatatuneException(f"Failed to select columns in view '{self.name}'.")
        return response
