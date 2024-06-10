from datatune.api import API
from datatune.entity import Entity
from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from datatune.workspace import Workspace  # This import is only processed by type checkers and never at runtime.


class Dataset:
    def __init__(self, id: str, workspace: Workspace):
        self.id = id
        self.workspace = workspace
    
    @property
    def entity(self) -> Entity:
        return self.workspace.entity

    @property
    def name(self) -> str:
        return self.workspace.api.get_dataset(self.id, entity=self.workspace.entity.id, workspace=self.workspace.id)["name"]
    
    @property
    def api(self) -> API:
        return self.workspace.api

class DatasetSlice:
    def __init__(self, dataset: Dataset, start: Optional[int] = None, end: Optional[int] = None):
        self.dataset = dataset
        self.start = start
        self.end = end
