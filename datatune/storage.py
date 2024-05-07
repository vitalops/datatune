from .api import API
from .exceptions import DatatuneException
from .dataset import Dataset
from .config import DATATUNE_STORAGE_API_BASE_URL
import os


class Storage:
    """
    Handles storage operations for datasets within the Datatune platform.
    """

    def __init__(self, api_key):
        self.api = API(api_key=api_key,
                       base_url=DATATUNE_STORAGE_API_BASE_URL)

    def upload_dataset(self, name, path, is_local=False):
        """
        Uploads a dataset to the Datatune platform, handling local and cloud storage.

        Args:
            name (str): The dataset name.
            path (str): Path to the dataset.
            is_local (bool): Specifies if the dataset is stored locally.
        """
        if is_local:
            try:
                with open(path, 'rb') as file_data:
                    files = {'file': (os.path.basename(path), file_data)}
                    response = self.api.post('datasets/upload',
                                             files=files,
                                             data={'name': name})
            except Exception as e:
                raise DatatuneException(f"Failed to load local data and add dataset '{name}': {str(e)}")
        else:
            response = self.api.post('datasets/upload',
                                     json={'name': name, 'path': path})
    
        if not response.get('success'):
            raise DatatuneException("Failed to upload dataset.")
        return Dataset(self.api, name)

    def load_dataset(self, name):
        """
        Fetches a dataset by its name from the storage if it exists.
        """
        existing_datasets = self.list_datasets()
        if name not in existing_datasets:
            raise DatatuneException(f"Dataset '{name}' does not exist.")
        return Dataset(self.api, name)

    def download_dataset(self, name, save_path):
        """
        Downloads a dataset from the Datatune platform.

        Args:
            name (str): The name of the dataset to download.
            save_path (str): The local path to save the dataset.
        """
        response = self.api.get(f'datasets/{name}/download')
        if response.get('success'):
            with open(save_path, 'wb') as f:
                f.write(response['data'])
        else:
            raise DatatuneException("Failed to download dataset.")

        return Dataset(self.api, name)

    def delete_dataset(self, name):
        """
        Deletes a dataset from the Datatune platform.

        Args:
            name (str): The name of the dataset to delete.
        """
        response = self.api.delete(f'datasets/{name}')
        if not response.get('success'):
            raise DatatuneException("Failed to delete dataset.")
        return self

    def list_datasets(self):
        """
        Lists all datasets available in the Datatune platform.

        Returns:
            list: A list of dataset names.
        """
        response = self.api.get('datasets')
        if not response.get('success'):
            raise DatatuneException("Failed to list datasets.")
        return [dataset['name'] for dataset in response.get('data', [])]
