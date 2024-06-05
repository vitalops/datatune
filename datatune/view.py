from datatune.api import API
from datatune.workspace import Workspace
from datatune.entity import Entity
from datatune.dataset import Dataset, DatasetSlice
from typing import List, Optional, Tuple


class View:
    def __init__(self, id: str, workspace: Workspace):
        self.id = id
        self.workspace = workspace
    
    @property
    def entity(self) -> Entity:
        return self.workspace.entity

    @property
    def name(self) -> str:
        return self.workspace.api.get_view(self.id, entity=self.workspace.entity.id, workspace=self.workspace.id)["name"]
    
    @property
    def api(self) -> API:
        return self.workspace.api

    @property
    def dataset_slices(self) -> List[DatasetSlice]:
        from datatune.dataset import DatasetSlice
        slices: List[Tuple[str, Tuple[Optional[int], Optional[int]]]] = self.api.get_view(self.id, entity=self.entity.id, workspace=self.workspace.id)["dataset_slices"]
        return [DatasetSlice(dataset=Dataset(id=dataset_id, workspace=self.workspace), start=start, end=end) for dataset_id, (start, end) in slices]
    
    @property
    def extra_columns(self) -> List:
        from datatune.extra_column import ExtraColumn
        column_ids = self.api.list_extra_columns(entity=self.entity.id, workspace=self.workspace.id, view=self.id)
        return [ExtraColumn(id=column_id, view=self) for column_id in column_ids]
