from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Union

class BaseStorage(ABC):
    """
    Base class for handling file and data operations across different storage backends,
    suitable for a data lakehouse architecture.
    """

    def __init__(self, config):
        """
        Initializes the storage with necessary configuration.
        Parameters:
            config (dict): Configuration dictionary containing settings like access keys, secrets, and region.
        """
        self.config = config

    # Dictionary-like methods
    def __getitem__(self, key: str) -> bytes:
        return self._get(key)

    def __setitem__(self, key: Union[str, List[str]], value: bytes) -> None:
        if isinstance(key, list):
            self._pset(key, value)
        else:
            self._set(key, value)

    def __delitem__(self, key: str) -> None:
        self.delete_file(key)

    def __contains__(self, key: str) -> bool:
        return self.file_exists(key)

    def __len__(self) -> int:
        return self.count_files()

    @abstractmethod
    def upload_file(self, file_path: str, destination: str) -> None:
        pass

    @abstractmethod
    def download_file(self, source: str, destination: str) -> None:
        pass

    @abstractmethod
    def list_files(self, path: str) -> Iterator[str]:
        pass

    @abstractmethod
    def delete_file(self, path: str) -> None:
        pass

    @abstractmethod
    def get_metadata(self, path: str) -> dict:
        pass

    @abstractmethod
    def set_metadata(self, path: str, metadata: dict) -> None:
        pass

    @abstractmethod
    def query_data(self, query: str, options: Optional[dict] = None) -> Iterator[dict]:
        pass

    # Private methods to be implemented by subclasses
    @abstractmethod
    def _get(self, key: str) -> bytes:
        pass

    @abstractmethod
    def _set(self, key: str, value: bytes) -> None:
        pass

    @abstractmethod
    def _pset(self, keys: List[str], values: List[bytes]) -> None:
        pass

    @abstractmethod
    def _pget(self, keys: List[str]) -> List[bytes]:
        pass

    # Example utility methods
    def file_exists(self, path: str) -> bool:
        try:
            self.get_metadata(path)
            return True
        except FileNotFoundError:
            return False

    def count_files(self, directory: str = '') -> int:
        return sum(1 for _ in self.list_files(directory))

    def __str__(self):
        return f"{self.__class__.__name__} using config: {self.config}"
