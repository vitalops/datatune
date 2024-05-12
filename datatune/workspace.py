from .api import API
from .exceptions import DatatuneException
from .view import View
from .dataset import Dataset
import requests


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
    """


    def __init__(self, uri, token):
        self.uri = uri
        self.token = token
        self.api = API(api_key=token)
        self.user, self.workspace_name = self._parse_uri(uri)

    def add_credentials(self, config):
        """
        Adds cloud storage credentials to the workspace for data handling.
        
        Args:
            config (dict): Configuration dictionary containing credentials for cloud storage.
                Expected keys:
                - 'aws_access_key_id'
                - 'aws_secret_access_key'
                - 'google_cloud_key'
                - 'azure_storage_key'
                - 'huggingface_token'
        """
        self.credentials = config

    def add_dataset(self, dataset_id, path, is_local=False):
        """
        Adds a dataset to the workspace, handling both local and cloud data.

        Args:
            dataset_id (str): Unique identifier of the dataset.
            path (str): Path or URL to the dataset file.
            is_local (bool): Flag to indicate if the dataset is stored locally.
        """
        if is_local:
            with open(path, 'rb') as file:
                response = self.api.add_dataset(self.workspace_name,
                                                dataset_id,
                                                path,
                                                is_local,
                                                self.credentials,
                                                file)
        else:
            response = self.api.add_dataset(self.workspace_name,
                                            dataset_id,
                                            path,
                                            is_local,
                                            self.credentials)

        if not response.get('success'):
            raise DatatuneException("Failed to add dataset.")
        return Dataset(dataset_id,
                       api=self.api,
                       workspace=self)

    def load_dataset(self, dataset_id):
        """
        Fetches a dataset by its name from the workspace if it exists.
        """
        existing_datasets = self.list_datasets()
        if dataset_id not in [dataset.dataset_id for dataset in existing_datasets]:
            raise DatatuneException(f"Dataset '{dataset_id}' does not exist.")
        return Dataset(dataset_id,
                       api=self.api,
                       workspace=self)

    def delete_dataset(self, dataset_id):
        """
        Deletes a dataset from the workspace.
        """
        response = self.api.delete_dataset(self.workspace_name, dataset_id)
        if not response.get('success'):
            raise DatatuneException("Failed to delete dataset.")
        return self

    def list_datasets(self):
        """
        Returns a list of all datasets in the workspace.
        """
        response = self.api.list_datasets(self.workspace_name)
        if not response.get('success'):
            raise DatatuneException("Failed to list datasets.")
        datasets = [Dataset(ds['dataset_id'],
                            api=self.api,
                            workspace=self) for ds in response.get('datasets',
                                                                   [])]
        return datasets

    def create_view(self, view_name):
        """
        Creates a new view in the workspace.

        Args:
            view_name (str): Name of the view to create.
        """
        response = self.api.create_view(self.workspace_name, view_name)
        if not response.get('success'):
            raise DatatuneException("Failed to create view.")
        return View(self, view_name)

    def load_view(self, view_name):
        """
        Fetches a view by its name from the workspace.

        Args:
            view_name (str): Name of the view to fetch.
        """
        response = self.api.load_view(self.workspace_name, view_name)
        if not response.get('success'):
            raise DatatuneException(f"Failed to load view '{view_name}'.")
        return View(self, view_name)

    def delete_view(self, view_name):
        """
        Deletes a view from the workspace.

        Args:
            view_name (str): Name of the view to delete.
        """
        response = self.api.delete_view(self.workspace_name, view_name)
        if not response.get('success'):
            raise DatatuneException("Failed to delete view.")
        return self

    def list_views(self):
        """
        Lists all views in the workspace.
        """
        response = self.api.list_views(self.workspace_name)
        if not response.get('success'):
            raise DatatuneException("Failed to list views.")
        return [View(self, view['name']) for view in response.get('views', [])]

    def _parse_uri(self, uri):
        scheme, path = uri.split("://")
        user, workspace_name = path.split("/")
        return user, workspace_name
