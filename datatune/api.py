import platform
import requests
from requests.adapters import HTTPAdapter, Retry
from .exceptions import DatatuneException
from .config import DATATUNE_API_BASE_URL
from .constants import (
    HTTP_RETRY_BACKOFF_FACTOR,
    HTTP_STATUS_FORCE_LIST,
    HTTP_TOTAL_RETRIES,
)
from typing import Optional, Dict, List, Tuple, Any, Union, Generator
import logging
import json


logging.basicConfig(level=logging.INFO)


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
        if response.status_code not in [200, 201]:
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
        workspace: str,
        path: Union[str, List[str]],
        credentials: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> str:

        json_payload = {
            "name": name,
            "paths": path, 
            "description": description if description else "No description provided",
        }

        if credentials is not None:
            json_payload["credentials_id"] = credentials
        
        if dataset_type is not None:
            json_payload["dataset_type"] = dataset_type

        resp = self.post(
            endpoint=f"workspaces/{workspace}/datasets",
            json=json_payload,
        )["data"]
        return resp['id']

    def delete_dataset(self, entity: str, workspace: str, dataset: str) -> None:
        self.delete(
            endpoint=f"workspaces/{workspace}/datasets/{dataset}",
        )

    def list_datasets(self, workspace: str) -> List[str]:
        response = self.get(
            endpoint=f"workspaces/{workspace}/datasets",
            params={
                "workspace_id": workspace,
            },
        )["data"]
        ids = [dataset["id"] for dataset in response]
        return ids

    def list_workspaces(self, entity: str) -> List[str]:
        response = self.get(
            endpoint="workspaces",
            params={
                "entity": entity,
            },
        )
        response = response["data"]
        ids = [workspace["id"] for workspace in response]
        return ids
    

    def create_workspace(
        self,
        entity: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        response = self.post(
            endpoint=f"organizations/{entity}/workspaces",
            json={
                "name": name,
                "description": description
            },
        )
        return response["data"]["id"]

    def update_workspace(
        self, id: str, name: Optional[str] = None, description: Optional[str] = None
    ) -> str:
        response = self.put(
            endpoint=f"workspaces/{id}", json={"name": name, "description": description}
        )
        return response

    def list_extra_columns(self, entity: str, workspace: str, view: str) -> List[str]:
        response = self.get(
            endpoint="columns",
            params={
                "organization_id": entity,
                "workspace_id": workspace,
                "dataset_view_id": view,
            },
        )["data"]
        ids = [column["id"] for column in response]
        return ids

    def delete_workspace(self, entity: str, workspace: str) -> None:
        self.delete(
            endpoint=f"organizations/{entity}/workspaces/{workspace}",
        )

    def create_view(
        self,
        entity: str,
        workspace: str,
        view_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        return self.post(
            endpoint=f"workspaces/{workspace}/views",
            json={
                "name": view_name,
                "description": description
                if description
                else "No description provided",
            },
        )["data"]["id"]

    def delete_view(self, entity: str, workspace: str, view: str) -> None:
        self.delete(
            endpoint=f"workspaces/{workspace}/views/{view}",
        )

    def list_views(self, workspace: str) -> List[str]:
        response = self.get(
            endpoint=f"workspaces/{workspace}/views",
        )["data"]
        ids = [view["id"] for view in response]
        return ids

    def get_view(
        self,
        id: str,
    ) -> Dict:
        return self.get(endpoint=f"views/{id}")["data"]

    def get_dataset(self, workspace_id: str, id: str) -> Dict:
        return self.get(endpoint=f"workspaces/{workspace_id}/datasets/{id}")["data"]

    def get_extra_column(self, view: str) -> Dict:
        return self.get(endpoint=f"columns/{view}")["data"]

    def get_entity(self, id: str) -> Dict:
        return self.get(
            endpoint="get_entity",
            params={
                "id": id,
            },
        )["data"]

    def get_workspace(self, id: str) -> Dict:
        return self.get(
            endpoint=f"workspaces/{id}",
            params={
                "id": id,
            },
        )["data"]

    def extend_view(
        self,
        view: str,
        dataset: str,
        table: str,
        column_maps: Optional[Dict[str, str]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> None:

        slice = {"dataset": dataset}

        if start is not None:
            slice["start"] = start
        if stop is not None:
            slice["stop"] = stop

        slice['table'] = table
        json_payload = {"slices": [slice]}
    
        if column_maps is not None:
            json_payload['column_maps'] = column_maps

        

        response = self.put(endpoint=f"views/{view}/extend", json=json_payload)
        return response["data"]

    def add_extra_column(
        self,
        entity: str,
        workspace: str,
        view: str,
        column_name: str,
        column_type: str,  # one of "int", "float", "str", "bool", "label"
        labels: Optional[List[str]] = None,
        default_value: Any = None,
        num_rows: int=0,
        description: Optional[str] = None,
    ) -> str:
        response = self.post(
            endpoint="columns",
            json={
                "name": column_name,
                "description": description
                if description
                else "No description provided",
                "column_type": column_type,
                "default_value": default_value,
                "labels": labels,
                "organization_id": entity,
                "workspace_id": workspace,
                'num_rows':num_rows,
                "dataset_view_id": view,
            },
        )["data"][0] #temporary, will change this to add multiple columns
        return response["id"]

    def delete_extra_column(self, id: str) -> None:
        """Delete a column."""
        return self.delete(f"columns/{id}")

    def create_credentials(
        self,
        entity: str,
        workspace: str,
        name: str,
        credential_type: str,
        creds_details: Dict,
        path: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict:
        """Create a credential in a specific workspace."""
        json_payload = {
            "name": name,
            "description": description if description else "No description provided",
            "type": credential_type,
            "credentials": creds_details,
        }
        if path is not None:
            json_payload["path"] = path

        response = self.post(f"workspaces/{workspace}/credentials", json=json_payload)
        return response

    def get_credentials(self, credential_id: str) -> Dict:
        """Retrieve a specific credential by ID within a workspace."""
        return self.get(f"credentials/{credential_id}")["data"]

    def list_credentials(self, workspace: str) -> List[str]:
        response = self.get(
            endpoint=f"workspaces/{workspace}/credentials",
            params={
                "workspace_id": workspace,
            },
        )

        response = response["data"]
        ids = [credentials["id"] for credentials in response]
        return ids

    def delete_credential(self, workspace_id: str, credential_id: str) -> None:
        """Delete a specific credential."""
        return self.delete(f"workspaces/{workspace_id}/credentials/{credential_id}")

    def get_dataset_view_details(self, workspace_id: str, view_id: str) -> Dict:
        endpoint = f"workspaces/{workspace_id}/views/{view_id}/details"
        response = self.get(endpoint)
        return response["data"]

    def get_batches(
        self,
        workspace_id: str,
        view_id: str,
        start_index: int = 0,
        batch_size: int = 100,
        num_batches: int = 10
    ) -> Generator[Dict, None, None]:
        endpoint = f"workspaces/{workspace_id}/views/{view_id}/stream/batches"
        params = {
            "start_index": start_index,
            "batch_size": batch_size,
            "num_batches": num_batches
        }

        response = self.session.get(
            f"{self.base_url}/{endpoint}",
            params=params,
            stream=True
        )

        for line in response.iter_lines():
            if line:
                response_data = json.loads(line)
                if isinstance(response_data, dict) and 'data' in response_data:
                    yield response_data['data']
                else:
                    raise DatatuneException("Unexpected response format", 500)

    @staticmethod
    def generate_user_agent() -> str:
        """Generate a user agent string with details about the platform."""
        return f"Datatune/{platform.system()} {platform.release()} Python/{platform.python_version()}"

    def get_view_by_name(self, workspace_id: str, name: str) -> str:
        response = self.get(
            endpoint=f"workspaces/{workspace_id}/views/by-name/{name}",
        )
        return response["data"]["id"]