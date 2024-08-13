from datatune.api import API
from datatune.entity import Entity
from typing import Optional, TYPE_CHECKING, List
from datatune.workspace import Workspace


class Dataset:
    def __init__(self, id: str, workspace: Workspace):
        self.id = id
        self.workspace = workspace
        tables = self.workspace.api.get_dataset(self.workspace.id, self.id)['tables']
        self.tables =  tables
        

    @property
    def entity(self) -> Entity:
        return self.workspace.entity

    @property
    def name(self) -> str:
        return self.workspace.api.get_dataset(self.workspace.id, self.id)["name"]

    @property
    def api(self) -> API:
        return self.workspace.api


class DatasetSlice:
    def __init__(
        self, dataset: Dataset, table_index: Optional[int] = 0, start: Optional[int] = None, stop: Optional[int] = None
    ):
        self.dataset = dataset
        self.table_index = table_index
        self.start = start
        self.stop = stop
        self.table_id = self.dataset.tables[table_index]['id']