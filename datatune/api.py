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

    @staticmethod
    def generate_user_agent() -> str:
        """Generate a user agent string with details about the platform."""
        return f"Datatune/{platform.system()} {platform.release()} Python/{platform.python_version()}"

    @staticmethod
    def quote_string(text):
        """Safely quote strings to be used in URL paths."""
        return urllib.parse.quote(text, safe='')
