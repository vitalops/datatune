from datatune import Storage

# Initialize the storage with an API key
api_key = "your_api_key_here"
storage = Storage(api_key=api_key)

# Upload a dataset from a local file
local_dataset_path = "/path/to/your/local/dataset.csv"
dataset_name = "local_dataset"
try:
    dataset = storage.upload_dataset(name=dataset_name, path=local_dataset_path, is_local=True)
    print(f"Dataset {dataset_name} uploaded successfully.")
except Exception as e:
    print(f"Error uploading dataset: {e}")

# Download a dataset
try:
    download_path = "/path/to/your/download/location/dataset.csv"
    storage.download_dataset(name=dataset_name, save_path=download_path)
    print(f"Dataset {dataset_name} downloaded successfully to {download_path}.")
except Exception as e:
    print(f"Error downloading dataset: {e}")

# List all datasets
try:
    datasets = storage.list_datasets()
    print("Available datasets:", datasets)
except Exception as e:
    print(f"Error listing datasets: {e}")

# Delete a dataset
try:
    storage.delete_dataset(name=dataset_name)
    print(f"Dataset {dataset_name} deleted successfully.")
except Exception as e:
    print(f"Error deleting dataset: {e}")
