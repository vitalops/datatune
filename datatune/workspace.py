from datatune.api import API
from datatune.entity import Entity
from typing import List

class Workspace:
    def __init__(self, id: str, entity: Entity):
        self.id = id
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
