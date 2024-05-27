"""Contains constants that are used throughout the datatune platform"""

# Retry parameters
HTTP_TOTAL_RETRIES = 3
HTTP_RETRY_BACKOFF_FACTOR = 2
HTTP_STATUS_FORCE_LIST = [408, 429, 500, 502, 503, 504]
