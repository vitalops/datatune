from datatune import Storage, Stream
import torch

# Initialize the storage with an API key
api_key = "your_api_key_here"
storage = Storage(api_key=api_key)

# Upload a dataset from a local file
local_dataset_path = "/path/to/your/local/dataset.csv"
dataset_name = "local_dataset"
dataset = storage.upload_dataset(name=dataset_name, path=local_dataset_path, is_local=True)
print(f"Dataset {dataset_name} uploaded successfully.")

# Download a dataset
download_path = "/path/to/your/download/location/dataset.csv"
storage.download_dataset(name=dataset_name, save_path=download_path)
print(f"Dataset {dataset_name} downloaded successfully to {download_path}.")

# Load a dataset example
loaded_dataset = storage.load_dataset(name=dataset_name)
print(f"Dataset {dataset_name} loaded successfully.")

# List all datasets
datasets = storage.list_datasets()
print("Available datasets:", [ds.name for ds in datasets])

# Create a stream instance using the dataset name
stream = Stream(source='storage_dataset1')
batches =  stream.stream_batches()

# Simple training function to emulate processing streamed batches
def train(batches):
    """
    Simple training loop to process streamed data.
    """
    for batch in batches:
        # Simulate a processing step, e.g., feeding data into a machine learning model
        print("Processing batch of size:", batch.size())

# Run the training function
train(stream)

# Delete a dataset
storage.delete_dataset(name=dataset_name)
print(f"Dataset {dataset_name} deleted successfully.")

# List available views in the workspace after deletion
updated_datasets = storage.list_datasets()
print("Updated datasets in the workplace:", [ds.name for ds in updated_datasets])
