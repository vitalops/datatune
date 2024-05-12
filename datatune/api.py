import os
import platform
import urllib.parse
import requests
from requests.adapters import HTTPAdapter, Retry
from .exceptions import DatatuneException
from .config import DATATUNE_API_BASE_URL
from .constants import HTTP_RETRY_BACKOFF_FACTOR, HTTP_STATUS_FORCE_LIST, HTTP_TOTAL_RETRIES 


class API:
    """Handles HTTP operations for the datatune system"""

    def __init__(self, api_key, base_url=None, verify_ssl=True, proxies=None, headers=None):
        if not api_key:
            raise DatatuneException("API key is required.")

        self.api_key = api_key
        self.base_url = base_url or DATATUNE_API_BASE_URL
        self.session = requests.Session()
        # Initialize with default headers
        default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.generate_user_agent(),
        }
        # Update with any custom headers passed during initialization
        if headers:
            default_headers.update(headers)

        self.session.headers.update(default_headers)
        self.session.verify = verify_ssl
        if proxies:
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

    def request(self, method, endpoint, params=None, json=None, files=None):
        """Send a HTTP request and handle response."""
        url = f"{self.base_url}/{endpoint}"
        response = self.session.request(method, url, params=params, json=json, files=files)
        if response.status_code != 200:
            self.raise_for_status(response)
        return response.json()

    def raise_for_status(self, response):
        """Raise specific exceptions based on the HTTP status code."""
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        error = DatatuneException(message, response.status_code)
        raise error

    def get(self, endpoint, params=None):
        """Wrapper for GET requests."""
        return self.request('GET', endpoint, params=params)

    def post(self, endpoint, json=None, files=None):
        """Wrapper for POST requests."""
        return self.request('POST', endpoint, json=json, files=files)

    def put(self, endpoint, json=None):
        """Wrapper for PUT requests."""
        return self.request('PUT', endpoint, json=json)

    def delete(self, endpoint, json=None):
        """Wrapper for DELETE requests."""
        return self.request('DELETE', endpoint, json=json)

    def get_presigned_url(self, action, dataset_name):
        """Request a presigned URL for operations like upload/download."""
        endpoint = f"presigned/{action}/{dataset_name}"
        response = self.session.get(f"{self.base_url}/{endpoint}")
        if response.status_code == 200:
            return response.json()['url']
        else:
            raise DatatuneException("Failed to obtain presigned URL.")

    def add_dataset(self, workspace_name, dataset_id, path, is_local,
                    credentials=None,
                    file=None):
        """Add a dataset to the workspace."""
        endpoint = f"workspaces/{workspace_name}/datasets"
        if is_local and file:
            files = {'file': file}
            data = {'dataset_id': dataset_id}
            return self.post(endpoint + "/upload", files=files, json=data)
        else:
            json_data = {
                'dataset_id': dataset_id,
                'path': path,
                'credentials': credentials
            }
            return self.post(endpoint, json=json_data)

    def delete_dataset(self, workspace_name, dataset_id):
        """Delete a dataset from the workspace."""
        endpoint = f"workspaces/{workspace_name}/datasets/{dataset_id}"
        return self.delete(endpoint)

    def list_datasets(self, workspace_name):
        """List all datasets in the workspace."""
        endpoint = f"workspaces/{workspace_name}/datasets"
        return self.get(endpoint)

    def create_view(self, workspace_name, view_name):
        """Create a new view in the workspace."""
        endpoint = f"workspaces/{workspace_name}/views"
        json_data = {'name': view_name}
        return self.post(endpoint, json=json_data)

    def delete_view(self, workspace_name, view_name):
        """Delete a view from the workspace."""
        endpoint = f"workspaces/{workspace_name}/views/{view_name}"
        return self.delete(endpoint)

    def list_views(self, workspace_name):
        """List all views in the workspace."""
        endpoint = f"workspaces/{workspace_name}/views"
        return self.get(endpoint)

    def load_view(self, workspace_name, view_name):
        """Fetch a view by its name from the workspace."""
        endpoint = f"workspaces/{workspace_name}/views/{view_name}"
        return self.get(endpoint)

    def extend_view(self,
                    workspace_name,
                    view_name,
                    dataset_name,
                    slice_range):
        """Extend a view with a dataset slice."""
        endpoint = f"workspaces/{workspace_name}/views/{view_name}/extend"
        json_data = {'dataset_name': dataset_name, 'slice_range': slice_range}
        return self.put(endpoint, json=json_data)

    def add_column_to_view(self,
                           workspace_name,
                           view_name,
                           column_name,
                           column_type,
                           default_value=None):
        """Add a single column to a view."""
        endpoint = f"workspaces/{workspace_name}/views/{view_name}/columns"
        json_data = {'column_name': column_name,
                     'column_type': column_type,
                     'default_value': default_value}
        return self.post(endpoint, json=json_data)

    def execute_query(self, workspace_name, view_name, query_str):
        """
        Execute a SQL query against a specified view in the workspace.
        """
        endpoint = f"workspaces/{workspace_name}/views/{view_name}/query"
        json_data = {'query': query_str}
        response = self.post(endpoint, json=json_data)
        return response

    @staticmethod
    def generate_user_agent() -> str:
        """Generate a user agent string with details about the platform."""
        return f"Datatune/{platform.system()} {platform.release()} Python/{platform.python_version()}"

    @staticmethod
    def quote_string(text):
        """Safely quote strings to be used in URL paths."""
        return urllib.parse.quote(text, safe='')
