import requests
import torch
from torch.utils.data import IterableDataset
from .exceptions import DatatuneException


class TorchStream(IterableDataset):
    def __init__(self, view, batch_size=1024):
        # Convert the view data to a PyTorch-friendly format before streaming
        view = view.convert_to_pytorch()
        self.view = view
        self.batch_size = batch_size

    def __iter__(self):
        return self.stream_batches()

    def stream_batches(self):
        session = requests.Session()
        endpoint = f"{self.view.workspace.api.base_url}/stream_pytorch/{self.view.name}"
        with session.get(endpoint, stream=True) as response:
            if response.status_code == 200:
                batch_data = []
                for line in response.iter_lines():
                    if line:
                        processed_data = self.process_data(line)
                        batch_data.append(processed_data)
                        if len(batch_data) >= self.batch_size:
                            yield torch.tensor(batch_data)
                            batch_data = []
                if batch_data:
                    yield torch.tensor(batch_data)  # Yield the last batch if there's remaining data
            else:
                raise DatatuneException("Failed to stream data.")

    def process_data(self, line):
        # Placeholder for data processing logic
        return line
