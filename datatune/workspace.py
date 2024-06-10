from datatune.api import API
from datatune.entity import Entity
from typing import List, Optional
from datatune.dataset import Dataset
from datatune.view import View


class Workspace:
    def __init__(self, id: str, entity: Entity):
        self.id = entity.api.create_workspace(entity.id, id)
        self.entity = entity

    @property
    def name(self) -> str:
        return self.entity.api.get_workspace(self.id, entity=self.entity.id)["name"]

    @property
    def views(self) -> List:
        from datatune.view import View
        view_ids =  self.entity.api.list_views(entity=self.entity.id, workspace=self.id)
        return [View(id=view_id, workspace=self) for view_id in view_ids]

    @property
    def datasets(self) -> List:
        from datatune.dataset import Dataset
        dataset_ids =  self.entity.api.list_datasets(entity=self.entity.id, workspace=self.id)
        return [Dataset(id=dataset_id, workspace=self) for dataset_id in dataset_ids]

    @property
    def api(self) -> API:
        return self.entity.api

    def delete(self):
        self.api.delete_workspace(entity=self.entity.id,
                                         workspace=self.id)
        self.id = None

    def add_dataset(self, path: str, creds_key: Optional[str]=None, name: Optional[str]=None) -> str:
        dataset_id = self.api.add_dataset(self.entity.id,
                                                 self.id,
                                                 path,
                                                 creds_key,
                                                 name)
        return Dataset(id=dataset_id, workspace=self)

    def load_dataset(self, id: str) -> Dataset:
        return Dataset(id=id, workspace=self)

    def delete_dataset(self, id: str) -> None:
        self.api.delete_dataset(self.entity.id, self.id, id)

    def create_view(self, view_name: str) -> View:
        view_id = self.api.create_view(self.entity.id,
                                       self.id,
                                       view_name)
        return View(id=view_id, workspace=self)

    def load_view(self, id: str) -> View:
        return View(id=id, workspace=self)

    def delete_view(self, view_name: str) -> None:
        self.api.delete_view(self.entity.id, self.id, view_name)
