from datatune.api import API
from typing import List
from enum import Enum


class EntityType(Enum):
    USER = "user"
    TEAM = "team"


class Entity:
    def __init__(self, id: str, api: API):
        self.id = id
        self.api = api

    @property
    def workspaces(self) -> List[str]:
        from datatune.workspace import Workspace

        workspace_ids = self.api.list_workspaces(self.id)
        return [
            Workspace(id=workspace_id, entity=self) for workspace_id in workspace_ids
        ]

    @property
    def name(self) -> str:
        return self.api.get_entity(self.id)["name"]

    @property
    def type(self) -> EntityType:
        return EntityType(self.api.get_entity(self.id)["type"])
