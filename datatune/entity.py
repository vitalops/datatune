from datatune.api import API
from typing import List
from enum import Enum


class EntityType(Enum):
    """
    Enumeration of possible entity types in Datatune.

    Attributes:
        USER: Represents an individual user entity
        TEAM: Represents a team entity
    """
    USER = "user"
    TEAM = "team"


class Entity:
    """
    A class representing an entity (user or team) in Datatune.

    Args:
        id (str): The unique identifier of the entity
        api (API): The API instance for making requests

    Attributes:
        id (str): The entity's unique identifier
        api (API): The API instance used for making requests
    """
    def __init__(self, id: str, api: API):
        self.id = id
        self.api = api

    @property
    def workspaces(self) -> List[str]:
        """
        Returns a list of workspaces associated with this entity.

        Returns:
            List[Workspace]: List of Workspace objects owned by this entity
        """
        from datatune.workspace import Workspace

        workspace_ids = self.api.list_workspaces(self.id)
        return [
            Workspace(id=workspace_id, entity=self) for workspace_id in workspace_ids
        ]

    @property
    def name(self) -> str:
        """
        Returns the name of the entity.

        Returns:
            str: The entity's name
        """
        return self.api.get_entity(self.id)["name"]

    @property
    def type(self) -> EntityType:
        """
        Returns the type of the entity.

        Returns:
            EntityType: The entity type (USER or TEAM)
        """
        return EntityType(self.api.get_entity(self.id)["type"])