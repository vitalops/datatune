from datatune.s3 import S3Storage


def main():
    # Configuration for the S3StorageConnector
    config = {
        'api_key': 'your_api_key_here',
        'datatune_storage_base_url': 'https://your.datatune.backend.url/api'
    }

    # Initialize the S3 storage connector
    s3_storage = S3Storage(config)

    # File paths
    local_file_to_upload = 'path/to/local/file.txt'
    destination_path_on_s3 = 'folder/on/s3/file.txt'
    download_path = 'path/to/downloaded/file.txt'

    # Upload a file
    print("Uploading file...")
    s3_storage.upload_file(local_file_to_upload, destination_path_on_s3)
    print("Upload complete.")

    # List files in a directory
    print("Listing files...")
    files = s3_storage.list_files('folder/on/s3/')
    for file in files:
        print(file)

    # Download a file
    print("Downloading file...")
    s3_storage.download_file(destination_path_on_s3, download_path)
    print(f"Download complete. File saved to {download_path}")

    # Get metadata of a file
    print("Fetching metadata...")
    metadata = s3_storage.get_metadata(destination_path_on_s3)
    print(f"Metadata: {metadata}")

    # Delete a file
    print("Deleting file...")
    s3_storage.delete_file(destination_path_on_s3)
    print("Delete complete.")

if __name__ == '__main__':
    main()
