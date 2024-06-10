from datatune.api import API
from datatune.workspace import Workspace
from datatune.entity import Entity
from datatune.dataset import Dataset, DatasetSlice
from typing import List, Optional, Tuple, Any
from datatune.extra_column import ExtraColumn


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

    def extend(self, dataset: Dataset, start: Optional[int] = None, end: Optional[int] = None) -> 'View':
        dataset_slice = DatasetSlice(dataset=dataset, start=start, end=end)
        self.api.extend_view(
            entity=self.entity.id,
            workspace=self.workspace.id,
            view=self.id,
            dataset=dataset_slice.dataset.id,
            range=(dataset_slice.start, dataset_slice.end)
        )
        return self

    def add_extra_column(self, column_name: str, column_type: str, labels: Optional[List[str]] = None, default_value: Any = None) -> 'View':
        self.api.add_extra_column(
            entity=self.entity.id,
            workspace=self.workspace.id,
            view=self.id,
            column_name=column_name,
            column_type=column_type,
            labels=labels,
            default_value=default_value
        )
        return self

    def load_extra_column(self, column_id: str) -> 'ExtraColumn':
        return ExtraColumn(id=column_id, view=self)
