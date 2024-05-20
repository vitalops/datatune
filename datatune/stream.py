from typing import Union
import s3fs
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader
from .exceptions import DatatuneException
from .storage import Storage
from .view import View
from .config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL


class Stream(IterableDataset):
    def __init__(self, source: Union[View, str], batch_size: int = 1024):
        """
        Initialize the dataset using a source object which could be a `view` instance or a `storage_dataset_id`.

        Args:
            source (Union[View, str]): A View instance or a storage dataset ID.
            batch_size (int): The number of items in each batch.

        Raises:
            DatatuneException: If neither a valid View instance nor a valid dataset ID is provided.
        """
        if isinstance(source, View):
            self.s3_path = source.get_remote_url()
        elif isinstance(source, str):
            dataset = Storage.load_dataset(source)
            self.s3_path = dataset.get_metadata()['remote_url']
        else:
            raise DatatuneException("Source must be either a View instance or a storage dataset ID.")

        self.fs = s3fs.S3FileSystem(
            key=AWS_ACCESS_KEY_ID,
            secret=AWS_SECRET_ACCESS_KEY,
            client_kwargs={'endpoint_url': S3_ENDPOINT_URL},
            use_ssl=False
        )
        self.parquet_file = pq.ParquetFile(self.s3_path, filesystem=self.fs)
        self.num_rows = self.parquet_file.metadata.num_rows
        self.batch_size = batch_size

    def __iter__(self):
        return self.stream_data()

    def stream_data(self):
        batch_iterator = self.parquet_file.iter_batches(batch_size=self.batch_size)
        for batch in batch_iterator:
            df = batch.to_pandas()
            processed_data = [self.process_data(row) for index, row in df.iterrows()]
            for item in processed_data:
                yield item

    def process_data(self, data):
        # Placeholder for data processing logic, this needs to be customized based on data format
        return data

    def __len__(self):
        # Return the total number of rows in the dataset
        return self.num_rows

    def get_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size)
