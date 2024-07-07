from datatune.api import API
from datatune.entity import Entity
from typing import Optional, TYPE_CHECKING
from datatune.workspace import Workspace


class Dataset:
    def __init__(self, id: str, workspace: Workspace):
        self.id = id
        self.workspace = workspace

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
        self, dataset: Dataset, start: Optional[int] = None, stop: Optional[int] = None
    ):
        self.dataset = dataset
        self.start = start
        self.stop = stop
