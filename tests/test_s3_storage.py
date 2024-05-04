import unittest
from unittest.mock import patch, MagicMock, ANY
from datatune.storage.s3 import S3Storage

class TestS3Storage(unittest.TestCase):
    def setUp(self):
        # Initialize API key and S3Storage instance
        self.api_key = 'test_api_key'
        with patch('datatune.storage.s3.API') as mock_api:
            self.mock_api_instance = mock_api.return_value
            self.storage = S3Storage(api_key=self.api_key)
        
        # Define the file paths used in the tests
        self.file_path = 'path/to/local/test.txt'
        self.destination = 'test/on/s3/test.txt'
        self.download_path = 'path/to/local/downloaded_test.txt'

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('datatune.storage.s3.API')
    def test_upload_file(self, mock_api, mock_file):
        # Setup the API and file mocks
        mock_api.return_value = self.mock_api_instance
        self.mock_api_instance.post.return_value = {'status': 'success'}
        
        # Test upload_file method
        self.storage.upload_file(self.file_path, self.destination)
        self.mock_api_instance.post.assert_called_once_with(
            'upload', 
            files={'file': (self.destination, ANY)}, 
            json={'destination': self.destination}
        )
        mock_file.assert_called_once_with(self.file_path, 'rb')

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('datatune.storage.s3.API')
    def test_download_file(self, mock_api, mock_file):
        # Setup the API and file mocks
        mock_api.return_value = self.mock_api_instance
        self.mock_api_instance.get.return_value = MagicMock(content=b'Test file content')
        
        # Test download_file method
        self.storage.download_file(self.destination, self.download_path)
        self.mock_api_instance.get.assert_called_once_with('download', params={'source': self.destination})
        mock_file.assert_called_once_with(self.download_path, 'wb')
        mock_file().write.assert_called_once_with(b'Test file content')

    # Additional tests for delete, list_files, etc.

if __name__ == '__main__':
    unittest.main()
