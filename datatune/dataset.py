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

    def add_credentials(self, config):
        """
        Adds cloud storage credentials to the workspace for data handling.
        
        Args:
            config (dict): Configuration dictionary containing credentials for cloud storage.
                Expected keys:
                - 'aws_access_key_id'
                - 'aws_secret_access_key'
                - 'google_cloud_key'
                - 'azure_storage_key'
                - 'huggingface_token'
        """
        self.credentials = config

    def add_data(self, path):
        """
        Loads the dataset from the specified storage path, supporting various 
        storage providers and local files using smart_open library.
        
        Args:
            path (str): The storage path of the dataset.
            
        Returns:
            DataFrame: The loaded data, format determined by file extension.
        """
        try:
            file_extension = path.split('.')[-1]

            if file_extension == 'csv':
                data = pd.read_csv(smart_open.open(path))
            elif file_extension == 'parquet':
                data = pd.read_parquet(smart_open.open(path))
            else:
                raise ValueError("Unsupported file format. Please use .csv or .parquet.")

            return data

        except Exception as e:
            raise DatatuneException(f"Failed to load data from {path}: {str(e)}")

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
