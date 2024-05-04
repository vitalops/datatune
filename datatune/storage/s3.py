from ..api import API
from .base import BaseStorage
from typing import Iterator


class S3Storage(BaseStorage):
    def __init__(self, config):
        super().__init__(config)
        self.api = API(api_key=config['api_key'],
                       base_url=config.get('DATATUNE_STORAGE_API_BASE_URL'))

    def upload_file(self, file_path: str, destination: str) -> None:
        with open(file_path, 'rb') as file_data:
            files = {'file': (destination, file_data)}
            self.api.post('upload', files=files,
                          json={'destination': destination})

    def download_file(self, source: str, destination: str) -> None:
        response = self.api.get('download', params={'source': source})
        with open(destination, 'wb') as file:
            file.write(response.content)

    def delete_file(self, path: str) -> None:
        self.api.delete('delete', json={'path': path})

    def list_files(self, path: str):
        response = self.api.get('list', params={'path': path})
        return response.json().get('files', [])

    def get_metadata(self, path: str) -> dict:
        response = self.api.get('metadata', params={'path': path})
        return response.json()

    def set_metadata(self, path: str, metadata: dict) -> None:
        self.api.put('metadata', json={'path': path, 'metadata': metadata})

    def _get(self, key: str) -> bytes:
        response = self.api.get('data', params={'key': key})
        return response.content

    def _set(self, key: str, value: bytes) -> None:
        self.api.post('data', files={'file': (key, value)})

    def _pset(self, keys: list[str], values: list[bytes]) -> None:
        raise NotImplementedError("Batch set operation is not supported.")

    def _pget(self, keys: list[str]) -> list[bytes]:
        raise NotImplementedError("Batch get operation is not supported.")

    def query_data(self, query: str, options: dict = None) -> Iterator[dict]:
        response = self.api.post('query',
                                 json={'query': query,
                                       'options': options})
        return iter(response.json().get('results', []))
