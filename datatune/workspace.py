from datatune.api import API
from typing import List, Optional, Dict, Union
from .exceptions import DatatuneException

class Workspace:
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
        self.api.delete_workspace(entity=self.entity.id, workspace=self.id)
        self.id = None

    def update(self, name: Optional[str] = None, description: Optional[str] = None):
        self.api.update_workspace(id=self.id, name=name, description=description)

    def add_dataset(self,
                     path: Union[str, List[str]],
                     name: Optional[str] = None, 
                     dataset_type: Optional[str] = None,
                     description: Optional[str] = None) -> str:
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
        from datatune.dataset import Dataset

        return Dataset(id=id, workspace=self)

    def delete_dataset(self, id: str) -> None:
        self.api.delete_dataset(self.entity.id, self.id, id)

    def create_view(self, view_name: str):
        from datatune.view import View

        view_id = self.api.create_view(self.entity.id, self.id, view_name)
        return View(id=view_id, workspace=self)

    def load_view(self, name: Union[str, 'View']) -> 'View':
        from datatune.view import View
        view_id = self.api.get_view_by_name(self.id, name)          
        return View(id=view_id, workspace=self)

    def delete_view(self, view_name: str) -> None:
        self.api.delete_view(self.entity.id, self.id, view_name)

    def add_credentials(
        self,
        name: str,
        credential_type: str,
        credentials: Dict,
        path: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.api.create_credentials(
            self.entity.id,
            self.id,
            name,
            credential_type,
            credentials,
            path,
            description,
        )
    
