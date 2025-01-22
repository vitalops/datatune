from datatune.api import API
from datatune.entity import Entity
from typing import Optional, TYPE_CHECKING, List
from datatune.workspace import Workspace


class Dataset:
    """
    A class representing a dataset in Datatune.

    Args:
        id (str): The unique identifier of the dataset
        workspace (Workspace): The workspace object that contains this dataset

    Attributes:
        id (str): The dataset's unique identifier
        workspace (Workspace): The workspace containing this dataset
        tables (List): List of tables in the dataset
    """
    def __init__(self, id: str, workspace: Workspace):
        self.id = id
        self.workspace = workspace
        tables = self.workspace.api.get_dataset(self.workspace.id, self.id)['tables']
        self.tables = tables

    @property
    def entity(self) -> Entity:
        """Returns the entity associated with this dataset's workspace."""
        return self.workspace.entity

    @property
    def name(self) -> str:
        """Returns the name of the dataset."""
        return self.workspace.api.get_dataset(self.workspace.id, self.id)["name"]

    @property
    def api(self) -> API:
        """Returns the API instance associated with this dataset's workspace."""
        return self.workspace.api


class DatasetSlice:
    """
    A class representing a slice or subset of a dataset.

    Args:
        dataset (Dataset): The dataset to slice
        table_index (Optional[int]): Index of the table in the dataset. Defaults to 0
        start (Optional[int]): Starting index of the slice. None means start from beginning
        stop (Optional[int]): Ending index of the slice. None means go until the end

    Attributes:
        dataset (Dataset): The dataset being sliced
        table_index (int): Index of the table in the dataset
        start (Optional[int]): Starting index of the slice
        stop (Optional[int]): Ending index of the slice
        table_id (str): ID of the table being sliced
    """
    def __init__(
        self, dataset: Dataset, table_index: Optional[int] = 0, start: Optional[int] = None, stop: Optional[int] = None
    ):
        self.dataset = dataset
        self.table_index = table_index
        self.start = start
        self.stop = stop
        self.table_id = self.dataset.tables[table_index]['id']