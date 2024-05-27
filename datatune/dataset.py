from .exceptions import DatatuneException

class Dataset:
    """
    Represents a dataset within the Datatune platform.

    Attributes:
        api (API): An instance of the API class for HTTP requests. If no workspace is provided,
                   this should be set directly.
        workspace (Optional[Workspace]): The workspace instance to which the dataset belongs, if any.
        name (str): The name of the dataset.
        id (str): The unique identifier of the dataset.
    """
    def __init__(self, dataset_id, api=None, workspace=None):
        self.dataset_id = dataset_id
        self.workspace = workspace
        if workspace:
            self.api = workspace.api
        elif api:
            self.api = api
        else:
            raise ValueError("API instance must be provided if workspace is not specified.")

    def get_metadata(self):
        """
        Retrieves metadata about the dataset, including dimensions, column names,
        types, or any other metadata stored in the dataset's catalogue.

        Returns:
            dict: A dictionary containing metadata information.
        """
        try:
            metadata = self.api.get(f"datasets/{self.dataset_id}/metadata")
            return metadata
        except Exception as e:
            raise DatatuneException(f"Failed to retrieve metadata for {self.dataset_id}: {str(e)}")
