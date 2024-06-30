from datatune.api import API
from datatune.entity import Entity

from enum import Enum
from typing import List
from datatune.workspace import Workspace
from datatune.view import View


class ColumnType(Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    LABEL = "label"


class ExtraColumn:
    def __init__(self, id: str, view: View):
        self.id = id
        self.view = view

    @property
    def entity(self) -> Entity:
        return self.view.entity

    @property
    def workspace(self) -> Workspace:
        return self.view.workspace

    @property
    def name(self) -> str:
        return self.view.api.get_extra_column(self.id)["name"]

    @property
    def type(self) -> ColumnType:
        return ColumnType(
            self.view.api.get_extra_column(
                self.id, entity=self.entity.id, workspace=self.workspace.id
            )["type"]
        )

    @property
    def api(self) -> API:
        return self.view.api

    @property
    def labels(self) -> List[str]:
        return self.view.api.get_extra_column(
            self.id, entity=self.entity.id, workspace=self.workspace.id
        )["labels"]
