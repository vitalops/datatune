from datatune.api import API
from typing import List, Optional, Dict, Union
from .exceptions import DatatuneException

class Workspace:
    """
    A class representing a workspace in Datatune for managing datasets, views, and credentials.

    Args:
        entity: The Entity object representing the organization
        id (Optional[str]): The workspace ID. If None, creates a new workspace
        name (Optional[str]): Name of the workspace. Defaults to "Awesome Workspace" if creating new
        description (Optional[str]): Description of the workspace. Defaults to "No description provided"
    """

    def __init__(self, entity, id: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None):
        if id is None:
            self.id = entity.api.create_workspace(entity.id,
                                                   name or "Awesome Workspace",
                                                   description or "No description provided")
        else:
            self.id = id

        self.entity = entity
        self.credentials_id = None

    @property
    def name(self) -> str:
        return self.entity.api.get_workspace(self.id)["name"]

    @property
    def views(self) -> List:
        from datatune.view import View

        view_ids = self.entity.api.list_views(workspace=self.id)
        return [View(id=view_id, workspace=self) for view_id in view_ids]

    @property
    def datasets(self) -> List:
        from datatune.dataset import Dataset

        dataset_ids = self.entity.api.list_datasets(workspace=self.id)
        return [Dataset(id=dataset_id, workspace=self) for dataset_id in dataset_ids]

    @property
    def credentials(self) -> List:
        from datatune.credentials import Credentials

        credentials_ids = self.entity.api.list_credentials(workspace=self.id)
        return [Credentials(id=id, workspace=self) for id in credentials_ids]

    @property
    def api(self) -> API:
        return self.entity.api

    def delete(self):
        """Deletes the current workspace."""
        self.api.delete_workspace(entity=self.entity.id, workspace=self.id)
        self.id = None

    def update(self, name: Optional[str] = None, description: Optional[str] = None):
        """
        Updates the workspace's name and/or description.

        Args:
            name (Optional[str]): New name for the workspace
            description (Optional[str]): New description for the workspace
        """
        self.api.update_workspace(id=self.id, name=name, description=description)

    def add_dataset(self,
                     path: Union[str, List[str]],
                     name: Optional[str] = None, 
                     dataset_type: Optional[str] = None,
                     description: Optional[str] = None) -> str:
        """
        Adds a new dataset to the workspace.

        Args:
            path (Union[str, List[str]]): Path or list of paths to the dataset files
            name (Optional[str]): Name of the dataset
            dataset_type (Optional[str]): Type of the dataset
            description (Optional[str]): Description of the dataset

        Returns:
            Dataset: A Dataset object representing the newly added dataset
        """
        from datatune.dataset import Dataset

        dataset_id = self.api.add_dataset(
            self.id,
            path, 
            self.credentials_id, 
            name, 
            description,
            dataset_type
        )

        return Dataset(id=dataset_id,  workspace=self)

    def load_dataset(self, id: str):
        """
        Loads an existing dataset by ID.

        Args:
            id (str): The ID of the dataset to load

        Returns:
            Dataset: The loaded Dataset object
        """
        from datatune.dataset import Dataset

        return Dataset(id=id, workspace=self)

    def delete_dataset(self, id: str) -> None:
        """
        Deletes a dataset from the workspace.

        Args:
            id (str): ID of the dataset to delete
        """
        self.api.delete_dataset(self.entity.id, self.id, id)

    def create_view(self, view_name: str):
        """
        Creates a new view in the workspace.

        Args:
            view_name (str): Name of the view to create

        Returns:
            View: A View object representing the newly created view
        """
        from datatune.view import View

        view_id = self.api.create_view(self.entity.id, self.id, view_name)
        return View(id=view_id, workspace=self)

    def load_view(self, name: Union[str, 'View']) -> 'View':
        """
        Loads an existing view by name.

        Args:
            name (Union[str, View]): Name of the view or View object to load

        Returns:
            View: The loaded View object
        """
        from datatune.view import View
        view_id = self.api.get_view_by_name(self.id, name)          
        return View(id=view_id, workspace=self)

    def delete_view(self, view_name: str) -> None:
        """
        Deletes a view from the workspace.

        Args:
            view_name (str): Name of the view to delete
        """
        self.api.delete_view(self.entity.id, self.id, view_name)

    def add_credentials(
        self,
        name: str,
        credential_type: str,
        credentials: Dict,
        path: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Adds new credentials to the workspace.

        Args:
            name (str): Name of the credentials
            credential_type (str): Type of credentials
            credentials (Dict): Dictionary containing credential information
            path (Optional[str]): Path associated with the credentials
            description (Optional[str]): Description of the credentials
        """
        self.api.create_credentials(
            self.entity.id,
            self.id,
            name,
            credential_type,
            credentials,
            path,
            description,
        )