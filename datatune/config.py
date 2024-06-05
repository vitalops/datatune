import os

# Load environment variables or set default values
DATATUNE_API_BASE_URL = os.getenv('DATATUNE_API_BASE_URL',
                                  'https://api.datatune.com/v1')

DATATUNE_STORAGE_API_BASE_URL = os.getenv('DATATUNE_STORAGE_API_BASE_URL',
                                  'https://api.datatune.com/v1/storage')

AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
S3_ENDPOINT_URL = os.environ['S3_ENDPOINT_URL']
