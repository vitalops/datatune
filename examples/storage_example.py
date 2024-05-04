from datatune.storage.s3 import S3Storage


def main():
    # Assuming API key and possibly custom configurations are stored securely or configured in the environment
    api_key = 'your_api_key_here'
    custom_config = {
        'aws_access_key_id': 'YOUR_ACCESS_KEY_ID',
        'aws_secret_access_key': 'YOUR_SECRET_ACCESS_KEY',
        'region': 'YOUR_REGION',
        'bucket_name': 'YOUR_BUCKET_NAME'
    }
    # Create an instance of S3Storage without custom config
    storage = S3Storage(api_key=api_key)
    
    # Perform operations
    file_path = 'path/to/local/file.txt'
    destination = 'folder/on/s3/file.txt'
    download_path = 'path/to/downloaded/file.txt'
    
    # Upload a file
    print("Uploading file...")
    storage.upload_file(file_path, destination)
    print("File uploaded.")

    # List files
    print("Listing files...")
    files = storage.list_files('folder/on/s3/')
    for file in files:
        print(file)
    
    # Download a file
    print("Downloading file...")
    storage.download_file(destination, download_path)
    print(f"Download complete. File saved to {download_path}")
    
    # Delete a file
    print("Deleting file...")
    storage.delete_file(destination)
    print("File deleted.")

    # Now create another instance with custom configuration
    storage_with_config = S3Storage(api_key=api_key, config=custom_config)
    
    # Upload a file with custom headers
    print("Uploading file with custom config...")
    storage_with_config.upload_file(file_path, destination)
    print("File uploaded with custom config.")


if __name__ == '__main__':
    main()
