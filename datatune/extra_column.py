from datatune.api import API
from datatune.entity import Entity

from enum import Enum
from typing import List
from datatune.workspace import Workspace
from datatune.view import View


class ColumnType(Enum):
    """
    Enumeration of supported data types for extra columns.

    Attributes:
        INT: Integer type
        FLOAT: Floating-point number type
        STR: String type
        BOOL: Boolean type
        LABEL: Categorical label type
    """
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    LABEL = "label"


class ExtraColumn:
    """
    A class representing an additional column added to a view in Datatune.

    Args:
        id (str): The unique identifier of the column
        view (View): The view object that contains this column

    Attributes:
        id (str): The column's unique identifier
        view (View): The view containing this column
    """
    def __init__(self, id: str, view: View):
        self.id = id
        self.view = view

    @property
    def entity(self) -> Entity:
        """Returns the entity associated with this column's view."""
        return self.view.entity

    @property
    def workspace(self) -> Workspace:
        """Returns the workspace containing this column's view."""
        return self.view.workspace

    @property
    def name(self) -> str:
        """
        Returns the name of the column.

        Returns:
            str: The column's name
        """
        return self.view.api.get_extra_column(self.id)["name"]

    @property
    def type(self) -> ColumnType:
        """
        Returns the data type of the column.

        Returns:
            ColumnType: The column's data type (INT, FLOAT, STR, BOOL, or LABEL)
        """
        return ColumnType(
            self.view.api.get_extra_column(
                self.id, entity=self.entity.id, workspace=self.workspace.id
            )["type"]
        )

    @property
    def api(self) -> API:
        """Returns the API instance associated with this column's view."""
        return self.view.api

    @property
    def labels(self) -> List[str]:
        """
        Returns the list of possible labels for categorical columns.

        Returns:
            List[str]: List of valid labels for this column if it's of type LABEL,
                      otherwise returns an empty list
        """
        return self.view.api.get_extra_column(
            self.id, entity=self.entity.id, workspace=self.workspace.id
        )["labels"]