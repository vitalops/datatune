import pandas as pd
import smart_open
from .exceptions import DatatuneException

class Dataset:
    """
    Represents a dataset within the Datatune platform.
    
    Provides methods to manage and interact with data for analysis or training.
    
    Attributes:
        api (API): An instance of the API class for HTTP requests.
        name (str): The name of the dataset.
    """
    def __init__(self, api, name):
        self.api = api
        self.name = name

    def get_metadata(self):
        """
        Retrieves metadata about the dataset, including dimensions, column names,
        types, or any other metadata stored in the dataset's catalogue.
        
        Returns:
            dict: A dictionary containing metadata information.
        """
        try:
            metadata = self.api.get(f"datasets/{self.name}/metadata")
            return metadata
        except Exception as e:
            raise DatatuneException(f"Failed to retrieve metadata for {self.name}: {str(e)}")
