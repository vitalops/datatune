import os
import platform
import urllib.parse
import requests
from requests.adapters import HTTPAdapter, Retry
from .exceptions import DatatuneException
from .config import DATATUNE_API_BASE_URL
from .constants import HTTP_RETRY_BACKOFF_FACTOR, HTTP_STATUS_FORCE_LIST, HTTP_TOTAL_RETRIES
from typing import Optional, Dict, List, Tuple, Any


class API:
    def __init__(self, api_key: str, base_url: Optional[str]=None, verify_ssl: bool=True, proxies: Optional[Dict] = None, headers: Optional[Dict] = None):
        self.api_key = api_key
        self.base_url = base_url or DATATUNE_API_BASE_URL
        self.session = requests.Session()
        default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.generate_user_agent(),
        }
        headers = headers or {}
        default_headers.update(headers)
        self.session.headers.update(default_headers)
        self.session.verify = verify_ssl
        proxies = proxies or {}
        self.session.proxies.update(proxies)
        retry_strategy = Retry(
            total=HTTP_TOTAL_RETRIES,
            backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
            status_forcelist=HTTP_STATUS_FORCE_LIST,
            allowed_methods=frozenset(['GET', 'POST', 'PUT', 'DELETE']),
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount('https://', adapter)

    def request(self, method: str, endpoint: str, params: Optional[Dict]=None, json: Optional[Dict]=None, files: Optional[Dict]=None) -> Dict:
        assert method in {'GET', 'POST', 'PUT', 'DELETE'}
        url = f"{self.base_url}/{endpoint}"
        response = self.session.request(method, url, params=params, json=json, files=files)
        if response.status_code != 200:
            self.handle_error(response)
        return response.json()

    def handle_error(self, response: requests.Response) -> None:
        """Raise specific exceptions based on the HTTP status code."""
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        error = DatatuneException(message, response.status_code)
        raise error

    def get(self, endpoint: str, params: Optional[Dict]=None) -> Dict:
        """Wrapper for GET requests."""
        return self.request('GET', endpoint, params=params)

    def post(self, endpoint: str, json: Optional[Dict]=None, files: Optional[Dict]=None) -> Dict:
        """Wrapper for POST requests."""
        return self.request('POST', endpoint, json=json, files=files)

    def put(self, endpoint: str, json: Optional[Dict]=None) -> Dict:
        """Wrapper for PUT requests."""
        return self.request('PUT', endpoint, json=json)

    def delete(self, endpoint: str, json: Optional[Dict]=None) -> Dict:
        """Wrapper for DELETE requests."""
        return self.request('DELETE', endpoint, json=json)


    def add_dataset(self, entity: str, workspace: str, path: str, creds_key: Optional[str]=None, name: Optional[str]=None) -> str:
        resp = self.get(
            endpoint="add_dataset",
            params={
                "entity": entity,
                "workspace": workspace,
                "path": path,
                "creds_key": creds_key,
                "name": name,
            }
        )
        return resp["dataset_id"]

    def delete_dataset(self, entity: str, workspace: str, dataset: str) -> None:
        self.get(
            endpoint="delete_dataset",
            params={
                "entity": entity,
                "workspace": workspace,
                "dataset": dataset,
            }
        )

    def list_datasets(self, entity: str, workspace: str) -> List[str]:
        return self.get(
            endpoint="list_datasets",
            params={
                "entity": entity,
                "workspace": workspace,
            }
        )["datasets"]


    def create_view(self, entity: str, workspace: str, view_name: Optional[str]=None):
        return self.get(
            endpoint="create_view",
            params={
                "entity": entity,
                "workspace": workspace,
                "view": view_name,
            }
        )["id"]

    def delete_view(self, entity: str, workspace: str, view: str) -> None:
        self.get(
            endpoint="delete_view",
            params={
                "entity": entity,
                "workspace": workspace,
                "view": view,
            }
        )

    def list_views(self, entity: str, workspace: str) -> List[str]:
        return self.get(
            endpoint="list_views",
            params={
                "entity": entity,
                "workspace": workspace,
            }
        )["views"]

    def get_view(self, entity: str, workspace: str, view: str) -> Dict:
        return self.get(
            endpoint="get_view",
            params={
                "entity": entity,
                "workspace": workspace,
                "view": view,
            }
        )

    def extend_view(self,
                    entity: str,
                    workspace: str,
                    view: str,
                    dataset: str,
                    range: Optional[Tuple[int, int]]=None) -> None:
        return self.get(
            endpoint="extend_view",
            params={
                "entity": entity,
                "workspace": workspace,
                "view": view,
                "dataset": dataset,
                "range": range,
            }
        )

    def add_column(self,
                            entity: str,
                            workspace: str,
                            view: str,
                            column_name: str,
                            column_type: str,
                            default_value: Any=None) -> str:
        return self.get(
            endpoint="add_column",
            params={
                "entity": entity,
                "workspace": workspace,
                "view": view,
                "column_name": column_name,
                "column_type": column_type,
                "default_value": default_value,
            }
        )["column_id"]


    @staticmethod
    def generate_user_agent() -> str:
        """Generate a user agent string with details about the platform."""
        return f"Datatune/{platform.system()} {platform.release()} Python/{platform.python_version()}"
