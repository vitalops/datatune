from datatune.api import API
from datatune.entity import Entity
from datatune.dataset import Dataset, DatasetSlice
from typing import List, Optional, Union, Tuple, Any, Dict
from datatune.workspace import Workspace


class View:
    def __init__(self, id: str, workspace: Workspace):
        self.id = id
        self.workspace = workspace

    @property
    def entity(self) -> Entity:
        return self.workspace.entity

    @property
    def name(self) -> str:
        return self.workspace.api.get_view(self.id)["name"]

    @property
    def api(self) -> API:
        return self.workspace.api

    @property
    def dataset_slices(self) -> List[DatasetSlice]:
        from datatune.dataset import DatasetSlice

        slices: List[
            Tuple[str, Tuple[Optional[int], Optional[int]]]
        ] = self.api.get_view(
            self.id, entity=self.entity.id, workspace=self.workspace.id
        )[
            "dataset_slices"
        ]
        return [
            DatasetSlice(
                dataset=Dataset(id=dataset_id, workspace=self.workspace),
                start=start,
                end=end,
            )
            for dataset_id, (start, end) in slices
        ]

    @property
    def extra_columns(self) -> List:
        from datatune.extra_column import ExtraColumn

        column_ids = self.api.list_extra_columns(
            entity=self.entity.id, workspace=self.workspace.id, view=self.id
        )
        return [ExtraColumn(id=column_id, view=self) for column_id in column_ids]

    def extend(
        self,
        data: Union[Dataset, DatasetSlice],
        table_index: Optional[int] = 0,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        column_maps: Optional[Dict[str, str]] = None,
    ) -> "View":

        if isinstance(data, Dataset):
            dataset_slice = DatasetSlice(dataset=data, table_index=table_index, start=start, stop=stop)
        elif isinstance(data, DatasetSlice):
            dataset_slice = data
            if start is not None:
                dataset_slice.start = start
            if stop is not None:
                dataset_slice.stop = stop
            dataset_slice.table_index = table_index
        else:
            raise TypeError("data must be either a Dataset or a DatasetSlice")

        self.api.extend_view(
            view=self.id,
            dataset=dataset_slice.dataset.id,
            table=dataset_slice.table_id,
            column_maps=column_maps,
            start=dataset_slice.start,
            stop=dataset_slice.stop,
        )
        return self

    def add_extra_column(
        self,
        column_name: str,
        column_type: str,
        labels: Optional[List[str]] = None,
        default_value: Any = None,
        num_rows:int=0
    ) -> "View":
        self.api.add_extra_column(
            entity=self.entity.id,
            workspace=self.workspace.id,
            view=self.id,
            column_name=column_name,
            column_type=column_type,
            labels=labels,
            default_value=default_value,
            num_rows=num_rows
        )
        return self

    def get_extra_column(self, id: str):
        from datatune.extra_column import ExtraColumn

        return ExtraColumn(id=id, view=self)

    def delete_extra_column(self, id: str):
        self.api.delete_extra_column(id)
