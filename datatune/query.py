from .exceptions import DatatuneException

class Query:
    """
    Represents an SQL query within the Datatune platform for a specific view.
    This class allows users to build and execute SQL queries dynamically based on method calls.

    Attributes:
        view (View): The view within the workspace on which this query will be executed.
    """

    def __init__(self, view):
        self.view = view
        self.api = view.workspace.api

    def select(self, columns="*"):
        if isinstance(columns, list):
            columns = ", ".join(columns)
        query_str = f"SELECT {columns} FROM {self.view.name}"
        res = self.execute(query_str)
        return res

    def filter(self, condition):
        query_str = f"SELECT * FROM {self.view.name} WHERE {condition}"
        res = self.execute(query_str)
        return res

    def sort(self, columns, ascending=True):
        order = 'ASC' if ascending else 'DESC'
        if isinstance(columns, list):
            columns = ", ".join(columns)
        query_str = f"SELECT * FROM {self.view.name} ORDER BY {columns} {order}"
        res = self.execute(query_str)
        return res

    def execute(self, query_str):
        workspace_name = self.view.workspace.workspace_name
        view_name = self.view.name
        return self.api.execute_query(workspace_name,
                                      view_name,
                                      query_str)
