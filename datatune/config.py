import os

# Load environment variables or set default values
DATATUNE_API_BASE_URL = os.getenv(
    "DATATUNE_API_BASE_URL", "https://api.datatune.com/v1"
)
