
# entity


## entity.EntityType


## entity.Entity


**Args:**

* **id (str)** - The unique identifier of the entity
* **api (API)** - The API instance for making requests
* **Attributes** - 
* **id (str)** - The entity's unique identifier
* **api (API)** - The API instance used for making requests

## entity.workspaces


**Returns:**

List[Workspace]: List of Workspace objects owned by this entity

## entity.name


**Returns:**

str: The entity's name

## entity.type


**Returns:**

EntityType: The entity type (USER or TEAM)