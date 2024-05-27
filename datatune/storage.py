import os
import requests
from typing import Dict, Optional, List
from .api import API
from .exceptions import DatatuneException
from .dataset import Dataset
from .config import DATATUNE_STORAGE_API_BASE_URL


class Storage:
    """
    Handles storage operations for datasets within the Datatune platform, supporting both local
    and external storage including AWS S3, GCP, Azure, and Hugging Face.
    """

    def __init__(self, api_key: str):
        self.api: API = API(api_key=api_key, base_url=DATATUNE_STORAGE_API_BASE_URL)
        self.credentials: Dict[str, Dict[str, str]] = {}

    def set_credentials(self, aws: Optional[Dict[str, str]] = None, 
                        gcp: Optional[Dict[str, str]] = None, 
                        azure: Optional[Dict[str, str]] = None, 
                        huggingface: Optional[str] = None) -> None:
        """
        Sets the credentials for accessing external storage services.

        Args:
            aws (dict): AWS credentials with access_key_id and secret_access_key.
            gcp (dict): GCP credentials with service_account_info JSON.
            azure (dict): Azure credentials with storage_account and storage_key.
            huggingface (str): Hugging Face API token.
        """
        if aws:
            self.credentials['aws'] = aws
        if gcp:
            self.credentials['gcp'] = gcp
        if azure:
            self.credentials['azure'] = azure
        if huggingface:
            self.credentials['huggingface'] = huggingface

    def upload_dataset(self, dataset_id: str, path: str, is_local: bool = False, source: Optional[str] = None) -> Dataset:
        """
        Uploads a dataset to the Datatune platform, handling both local and external sources.

        Args:
            dataset_id (str): The dataset id.
            path (str): Path to the dataset or URL for external datasets.
            is_local (bool): Specifies if the dataset is stored locally.
            source (str, optional): Source identifier ('aws', 'gcp', 'azure', 'huggingface') for external datasets.

        Returns:
            Dataset: An object representing the uploaded dataset.
        """
        if is_local:
            try:
                presigned_url = self.api.get_presigned_url('upload', dataset_id)
                with open(path, 'rb') as file_data:
                    files = {'file': (os.path.basename(path), file_data)}
                    response = requests.put(presigned_url, files=files)
                    response.raise_for_status()
            except Exception as e:
                raise DatatuneException(f"Failed to upload dataset '{dataset_id}': {str(e)}")
        else:
            credentials = self.credentials.get(source)
            if not credentials:
                raise DatatuneException("No credentials provided for the specified source.")

            response = self.api.upload_storage_dataset(dataset_id, path, is_local, credentials)
            if not response.get('success'):
                raise DatatuneException("Failed to upload dataset from external source.")

        return Dataset(dataset_id, self.api)

    def load_dataset(self, dataset_id: str) -> Dataset:
        """
        Fetches a dataset by its name from the workspace if it exists.
        """
        existing_datasets = self.list_datasets()
        if dataset_id not in [dataset.dataset_id for dataset in existing_datasets]:
            raise DatatuneException(f"Dataset '{dataset_id}' does not exist.")
        return Dataset(dataset_id, self.api)

    def download_dataset(self, dataset_id: str, save_path: str) -> Dataset:
        """
        Downloads a dataset from the Datatune platform.
        """
        try:
            presigned_url = self.api.get_presigned_url('download', dataset_id)
            response = requests.get(presigned_url)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            raise DatatuneException(f"Error downloading dataset '{dataset_id}': {str(e)}")

        return Dataset(dataset_id, self.api)

    def delete_dataset(self, dataset_id: str) -> 'Storage':
        """
        Deletes a dataset from the Datatune platform.
        """
        response = self.api.delete_storage_dataset(dataset_id)
        if not response.get('success'):
            raise DatatuneException("Failed to delete dataset.")
        return self

    def list_datasets(self) -> List[Dataset]:
        """
        Lists all datasets available in the Datatune platform.
        """
        response = self.api.list_storage_datasets()
        if not response.get('success'):
            raise DatatuneException("Failed to list datasets.")
        datasets = [Dataset(dataset['dataset_id'], self.api) for dataset in response.get('data', [])]
        return datasets
