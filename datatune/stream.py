from typing import Union
import torch
from torch.utils.data import IterableDataset, DataLoader
from mosaicml import StreamingDataset
from .exceptions import DatatuneException
from .storage import Storage
from .view import View


class Stream(IterableDataset):
    def __init__(self, source: Union[View, str], local_dir: str = '/tmp/', shuffle: bool = True, batch_size: int = 1024):
        """
        Initialize the dataset using a source object which could be a `view` instance or a `storage_dataset_id`.

        Args:
            source (Union[View, str]): A View instance or a storage dataset ID.
            local_dir (str): The local directory where dataset files are cached.
            shuffle (bool): Whether to shuffle the dataset.
            batch_size (int): The number of items in each batch.

        Raises:
            DatatuneException: If neither a valid View instance nor a valid dataset ID is provided.
        """
        if isinstance(source, View):
            self.remote_url = source.get_remote_url()
        elif isinstance(source, str):
            dataset = Storage.load_dataset(source)
            self.remote_url = dataset.get_metadata()['remote_url']
        else:
            raise DatatuneException("Source must be either a View instance or a storage dataset ID.")
        
        self.local_dir = local_dir
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        return self.stream_data()

    def stream_data(self):
        stream_dataset = StreamingDataset(
            remote=self.remote_url,
            local=self.local_dir,
            shuffle=self.shuffle
        )
        for data in stream_dataset:
            processed_data = self.process_data(data)
            yield processed_data

    def process_data(self, data):
        # Placeholder for data processing logic, this needs to be customized based on data format
        return data

    def get_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)


