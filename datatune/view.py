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

    def extend(self, dataset_name, slice_range):
        """Extends a view with a slice of a dataset."""
        data = {'dataset_name': dataset_name, 'slice_range': slice_range}
        response = self.workspace.api.put(f"workspaces/{self.workspace.workspace_name}/views/{self.name}",
                                          json=data)
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
                columns = data.columns.tolist()
                for column in columns:
                    self._add_column_to_view(column, data[column].dtype.name)
            elif isinstance(data, pd.Series):
                column_name = data.name if data.name else column_name
                if not column_name:
                    raise ValueError("Column name must be provided for unnamed Series.")
                self._add_column_to_view(column_name, data.dtype.name)
            elif isinstance(data, str) and data.endswith('.parquet'):
                try:
                    df = pd.read_parquet(data)
                    for column in df.columns:
                        self._add_column_to_view(column, df[column].dtype.name)
                except Exception as e:
                    raise DatatuneException(f"Failed to load data from {data}: {str(e)}")
            else:
                raise ValueError("Unsupported data input. Provide a DataFrame, Series, or a Parquet file path.")
        elif column_name:
            self._add_column_to_view(column_name, column_type, default_value)
        else:
            raise ValueError("Either provide a data source or column details.")
        return self

    def _add_column_to_view(self, name, type, default=None):
        """
        Helper function to add a single column to the view via the API.
        """
        data = {'column_name': name,
                'column_type': type,
                'default_value': default}

        response = self.workspace.api.post(
            f"workspaces/{self.workspace.workspace_name}/views/{self.name}/columns",
            json=data
        )

        if not response.get('success'):
            raise DatatuneException(f"Failed to add column '{name}' to view '{self.name}'.")

    def query(self, sql_query):
        """Executes an SQL query on the view using the Query class."""
        query_instance = Query(self)
        query_instance.execute(sql_query)
        return self

    def display(self, n=5):
        """
        Fetches and displays the latest state of the view, returning the top 'n' rows.
        
        Args:
            n (int): Number of top rows to return from the view. Default is 5.
        
        Returns:
            pd.DataFrame: A DataFrame containing the top n rows of the view.
        
        Raises:
            DatatuneException: If the query execution fails or returns an error.
        """
        sql_query = f"SELECT * FROM {self.name} LIMIT {n}"
        query_instance = Query(self)
        try:
            df =  query_instance.execute(sql_query, to_df=True)
        except DatatuneException as e:
            raise DatatuneException(f"Error displaying data from view '{self.name}': {str(e)}")

        return df

    def is_pytorch_converted(self):
        """
        Checks if the view's data is already converted to PyTorch format.

        Returns:
            bool: True if converted, False otherwise.
        """
        response = self.workspace.api.get(f"workspaces/{self.workspace.workspace_name}/views/{self.name}/is_pytorch_converted")
        if response.get('success'):
            return response.get('data', {}).get('is_pytorch', False)
        else:
            raise DatatuneException("Failed to check if view is converted to PyTorch.")

    def convert_to_pytorch(self):
        """
        Converts the view's data into a format suitable for PyTorch, making an API call only if not already converted.
        """
        if not self.is_pytorch_converted():
            response = self.workspace.api.post("convert_to_pytorch", json={'workspace_name': self.workspace.workspace_name, 'view_name': self.name})
            if not response.get('success'):
                raise DatatuneException(f"Failed to convert view '{self.name}' to PyTorch: {response.get('message', '')}")
        return self
