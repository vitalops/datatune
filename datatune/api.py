import os
import platform
import urllib.parse
import requests
from requests.adapters import HTTPAdapter, Retry
from .exceptions import DatatuneException
from .config import DATATUNE_API_BASE_URL
from .constants import (
    HTTP_RETRY_BACKOFF_FACTOR,
    HTTP_STATUS_FORCE_LIST,
    HTTP_TOTAL_RETRIES,
)
from typing import Optional, Dict, List, Tuple, Any


class API:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        verify_ssl: bool = True,
        proxies: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ):
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
            allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        files: Optional[Dict] = None,
    ) -> Dict:
        assert method in {"GET", "POST", "PUT", "DELETE"}
        url = f"{self.base_url}/{endpoint}"
        response = self.session.request(
            method, url, params=params, json=json, files=files
        )
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

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Wrapper for GET requests."""
        return self.request("GET", endpoint, params=params)

    def post(
        self, endpoint: str, json: Optional[Dict] = None, files: Optional[Dict] = None
    ) -> Dict:
        """Wrapper for POST requests."""
        return self.request("POST", endpoint, json=json, files=files)

    def put(self, endpoint: str, json: Optional[Dict] = None) -> Dict:
        """Wrapper for PUT requests."""
        return self.request("PUT", endpoint, json=json)

    def delete(self, endpoint: str, json: Optional[Dict] = None) -> Dict:
        """Wrapper for DELETE requests."""
        return self.request("DELETE", endpoint, json=json)

    def add_dataset(
        self,
        entity: str,
        workspace: str,
        path: str,
        credentials: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        namespace = f"{entity}-{name}-dataset".replace(' ', '-').lower()
        resp = self.post(
            endpoint=f'workspaces/{workspace}/datasets',
            json={
                "namespace": namespace,
                "name": name,
                "path": path,
                "credentials_id": credentials,
                "description": "Awesome Dataset",
            },
        )
        return resp["id"]

    def delete_dataset(self, entity: str, workspace: str, dataset: str) -> None:
        self.get(
            endpoint="delete_dataset",
            params={
                "entity": entity,
                "workspace": workspace,
                "dataset": dataset,
            },
        )

    def list_datasets(self, entity: str, workspace: str) -> List[str]:
        return self.get(
            endpoint="list_datasets",
            params={
                "entity": entity,
                "workspace": workspace,
            },
        )["datasets"]

    def list_workspaces(self, entity: str) -> List[str]:
        return self.get(
            endpoint="list_workspaces",
            params={
                "entity": entity,
            },
        )["workspaces"]

    def create_workspace(self, entity: str, name: str) -> str:
        namespace = f"{entity}-{name}-workspace".replace(' ', '-').lower()
        response = self.post(
            endpoint=f'organizations/{entity}/workspaces',
            json={
                "namespace": namespace,
                "name": f"{name} Workspace",
                "description": "Awesome Workspace"
            }
        )
        return response['id']

    def list_extra_columns(self, entity: str, workspace: str, view: str) -> List[str]:
        return self.get(
            endpoint="list_extra_columns",
            params={
                "entity": entity,
                "workspace": workspace,
                "view": view,
            },
        )["columns"]

    def delete_workspace(self, entity: str, workspace: str) -> None:
        raise Exception("Unsafe operation")
        self.get(
            endpoint="delete_workspace",
            params={
                "entity": entity,
                "workspace": workspace,
            },
        )

    def create_view(self, entity: str, workspace: str, view_name: Optional[str] = None):
        namespace = f"{entity}-{view_name}-view".replace(' ', '-').lower()
        return self.post(
            endpoint=f'workspaces/{workspace}/views',
            json={
                "namespace": namespace,
                "name": view_name,
                "description": "Awesome View",
            },
        )["id"]

    def delete_view(self, entity: str, workspace: str, view: str) -> None:
        self.get(
            endpoint="delete_view",
            params={
                "entity": entity,
                "workspace": workspace,
                "view": view,
            },
        )

    def list_views(self, entity: str, workspace: str) -> List[str]:
        return self.get(
            endpoint="list_views",
            params={
                "entity": entity,
                "workspace": workspace,
            },
        )["views"]

    def get_view(self, id: str, entity: str, workspace: str) -> Dict:
        return self.get(
            endpoint="get_view",
            params={
                "entity": entity,
                "workspace": workspace,
                "id": id,
            },
        )

    def get_dataset(self, id: str, entity: str, workspace: str) -> Dict:
        return self.get(
            endpoint="get_dataset",
            params={
                "entity": entity,
                "workspace": workspace,
                "id": id,
            },
        )

    def get_extra_column(self, id: str, entity: str, workspace: str, view: str) -> Dict:
        return self.get(
            endpoint="get_extra_column",
            params={
                "entity": entity,
                "workspace": workspace,
                "view": view,
                "id": id,
            },
        )

    def get_entity(self, id: str) -> Dict:
        return self.get(
            endpoint="get_entity",
            params={
                "id": id,
            },
        )

    def get_workspace(self, id: str) -> Dict:
        return self.get(
            endpoint=f"workspaces/{id}",
            params={
                "id": id,
            },
        )

    def extend_view(
        self,
        view: str,
        dataset: str,
        start: Optional[int] = None,
        stop: Optional[int] = None
    ) -> None:
        slice = {"dataset": dataset}

        if start is not None:
            slice['start'] = start
        if stop is not None:
            slice['stop'] = stop
        
        json_payload = {"slices": [slice]}

        response =  self.put(
            endpoint=f'views/{view}/extend',
            json=json_payload
        )
        return response

    def add_extra_column(
        self,
        entity: str,
        workspace: str,
        view: str,
        column_name: str,
        column_type: str,  # one of "int", "float", "str", "bool", "label"
        labels: Optional[List[str]] = None,
        default_value: Any = None,
    ) -> str:
        response =  self.post(
            endpoint="columns",
            json={
                "name": column_name,
                "description": "This column is awesome!",
                "column_type": column_type,
                "default_value": default_value,
                "labels": labels,
                "organization_id": entity,
                "workspace_id": workspace,
                "dataset_view_id": view
            },
        )
        return response['id']
    
    def create_credentials(
        self,
        entity: str,
        workspace: str,
        name: str,
        credential_type: str,
        creds_details: Dict,
        path : Optional[str] = None, 
        description: Optional[str] = None
    ) -> Dict:
        """Create a credential in a specific workspace."""
        namespace = f"{entity}-{name}-workspace".replace(' ', '-').lower()
        json_payload = {
            "namespace": namespace,
            "name": name,
            "description": description if description else "No description provided",
            "type": credential_type,
            "credentials": creds_details
        }
        if path is not None:
            json_payload['path'] = path
        response =  self.post(f'workspaces/{workspace}/credentials', json=json_payload)
        return response['id']

    def get_credential(
        self,
        workspace_id: str,
        credential_id: str
    ) -> Dict:
        """Retrieve a specific credential by ID within a workspace."""
        return self.get(f'workspaces/{workspace_id}/credentials/{credential_id}')

    def update_credential(
        self,
        workspace_id: str,
        credential_id: str,
        namespace: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        credential_type: Optional[str] = None,
        path: Optional[str] = None,
        creds_details: Optional[Dict] = None
    ) -> Dict:
        """Update a specific credential within a workspace."""
        json_payload = {
            "namespace": namespace,
            "name": name,
            "description": description,
            "type": credential_type,
            "path": path,
            "credentials": creds_details
        }
        return self.put(f'workspaces/{workspace_id}/credentials/{credential_id}', json=json_payload)

    def delete_credential(
        self,
        workspace_id: str,
        credential_id: str
    ) -> None:
        """Delete a specific credential."""
        return self.delete(f'workspaces/{workspace_id}/credentials/{credential_id}')

    @staticmethod
    def generate_user_agent() -> str:
        """Generate a user agent string with details about the platform."""
        return f"Datatune/{platform.system()} {platform.release()} Python/{platform.python_version()}"
