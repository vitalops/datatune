import pandas as pd
from .exceptions import DatatuneException

class Query:
    """
    Represents an SQL query within the Datatune platform for a specific view.
    
    This class allows users to execute SQL queries specifically for filtering, transforming,
    or retrieving data from a particular view within the Datatune platform.
    
    Attributes:
        view (View): The view within the workspace on which this query will be executed.
    """
    def __init__(self, view):
        self.view = view

    def execute(self, query):
        """
        Executes the SQL query against the specified view and returns the results as a pandas DataFrame.

        Args:
            query (str): The SQL query to be executed.

        Returns:
            pd.DataFrame: The result of the query, typically a DataFrame containing the queried data.

        Raises:
            DatatuneException: If the query fails or the server returns an error.
        """
        data = {
            'workspace_name': self.view.workspace.workspace_name,
            'view_name': self.view.name,
            'query': query
        }
        response = self.view.workspace.api.post(
            "execute_query",
            json=data
        )
        if not response.get('success'):
            raise DatatuneException(f"Failed to execute query on view '{self.view.name}': {response.get('message', '')}")

        # Assuming the API response returns the data in a format that pandas can read directly:
        try:
            return pd.DataFrame(response.get('data', []))
        except ValueError as e:
            raise DatatuneException(f"Failed to convert query results into DataFrame: {str(e)}")

