from .api import API
from .exceptions import DatatuneException
from .view import View
from .dataset import Dataset


class Workspace:
    """
    Manages datasets and views within a specific workspace on the Datatune platform.
    
    The Workspace class is responsible for creating, deleting,
    and listing both datasets and views.
    It encapsulates all interactions with the API necessary to manage these resources.
    
    Attributes:
        uri (str): The URI of the workspace, containing the user and workspace name.
        token (str): The authentication token used for API requests.
        api (API): An instance of the API class to handle HTTP requests.
        user (str): The username extracted from the URI.
        workspace_name (str): The name of the workspace extracted from the URI.
    
    Methods:
        add_dataset(name, path): Adds a dataset to the cloud workspace.
        delete_dataset(name): Removes a dataset from the cloud workspace.
        list_datasets(): Returns a list of all datasets in the workspace.
        create_view(name): Creates a new view and returns a corresponding View object.
        delete_view(view_name): Deletes a view from the workspace.
        list_views(): Lists all views in the workspace.
    """


    def __init__(self, uri, token):
        self.uri = uri
        self.token = token
        self.api = API(api_key=token)
        self.user, self.workspace_name = self._parse_uri(uri)

    def add_dataset(self, name, path):
        """Adds a dataset to the cloud."""
        data = {'name': name, 'path': path}
        response = self.api.post(f"workspaces/{self.workspace_name}/datasets",
                                 json=data)
        if not response.get('success'):
            raise DatatuneException("Failed to add dataset.")

        return self

    def load_dataset(self, name):
        """
        Fetches a dataset by its name from the workspace.
        """
        response = self.api.get(f"workspaces/{self.workspace_name}/datasets/{name}")
        if not response.get('success'):
            raise DatatuneException(f"Failed to load dataset '{name}'.")
        return Dataset(self.api, name, response.get('data', {}).get('path'))

    def delete_dataset(self, name):
        """Deletes a dataset from the cloud."""
        response = self.api.delete(f"workspaces/{self.workspace_name}/datasets/{name}")
        if not response.get('success'):
            raise DatatuneException("Failed to delete dataset.")
        return self

    def list_datasets(self):
        """Returns a list of all datasets in the workspace."""
        response = self.api.get(f"workspaces/{self.workspace_name}/datasets")
        if not response.get('success'):
            raise DatatuneException("Failed to list datasets.")
        datasets = [Dataset(self.api, ds['name'], ds['path']) for ds in response.get('datasets', [])]
        return datasets

    def create_view(self, name):
        """Creates a new view and returns a View object."""
        response = self.api.post(f"workspaces/{self.workspace_name}/views", json={'name': name})
        if not response.get('success'):
            raise DatatuneException("Failed to create view.")
        return self

    def load_view(self, name):
        """
        Fetches a view by its name from the workspace.
        """
        response = self.api.get(f"workspaces/{self.workspace_name}/views/{name}")
        if not response.get('success'):
            raise DatatuneException(f"Failed to load view '{name}'.")
        return View(self, name)

    def delete_view(self, view_name):
        """Deletes a view from the workspace."""
        response = self.api.delete(f"workspaces/{self.workspace_name}/views/{view_name}")
        if not response.get('success'):
            raise DatatuneException("Failed to delete view.")
        return self

    def list_views(self):
        """Lists all views in the workspace."""
        response = self.api.get(f"workspaces/{self.workspace_name}/views")
        if not response.get('success'):
            raise DatatuneException("Failed to list views.")
        views = [View(self, view['name']) for view in response.get('views', [])]
        return views

    def _parse_uri(self, uri):
        scheme, path = uri.split("://")
        user, workspace_name = path.split("/")
        return user, workspace_name
