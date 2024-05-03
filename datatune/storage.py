import os
from abc import ABC, abstractmethod
from typing import List, Iterator

class Storage(ABC):
    """
    Base class for handling file operations across different storage backends.
    This abstract class provides a template for uploading, downloading, listing, and deleting
    files or objects within cloud storage services like AWS S3, Google Cloud Storage, and Azure Blob Storage.
    """

    def __init__(self, config):
        """
        Initializes the storage with necessary configuration.
        
        Parameters:
            config (dict): Configuration dictionary containing settings like access keys, secrets, and region.
        """
        self.config = config

    @abstractmethod
    def upload_file(self, file_path: str, destination: str) -> None:
        """
        Uploads a file to the storage backend.

        Parameters:
            file_path (str): The local path to the file.
            destination (str): The destination path in the storage backend.
        """
        pass

    @abstractmethod
    def download_file(self, source: str, destination: str) -> None:
        """
        Downloads a file from the storage backend.

        Parameters:
            source (str): The path of the file in the storage backend.
            destination (str): The local destination path where the file will be saved.
        """
        pass

    @abstractmethod
    def list_files(self, path: str) -> Iterator[str]:
        """
        Lists all files under a directory in the storage backend.

        Parameters:
            path (str): The directory path in the storage backend from which to list files.

        Returns:
            Iterator[str]: An iterator over the filenames under the given path.
        """
        pass

    @abstractmethod
    def delete_file(self, path: str) -> None:
        """
        Deletes a file from the storage backend.

        Parameters:
            path (str): The path of the file to delete in the storage backend.
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__} using config: {self.config}"