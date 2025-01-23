# entity

## entity.EntityType

Enumeration of possible entity types in Datatune.

Attributes:
    USER: Represents an individual user entity
    TEAM: Represents a team entity

## entity.Entity

A class representing an entity (user or team) in Datatune.

Args:
    id (str): The unique identifier of the entity
    api (API): The API instance for making requests

Attributes:
    id (str): The entity's unique identifier
    api (API): The API instance used for making requests

## entity.workspaces

Returns a list of workspaces associated with this entity.

Returns:
    List[Workspace]: List of Workspace objects owned by this entity

## entity.name

Returns the name of the entity.

Returns:
    str: The entity's name

## entity.type

Returns the type of the entity.

Returns:
    EntityType: The entity type (USER or TEAM)

