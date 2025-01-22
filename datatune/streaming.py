import threading
import time
import math
from typing import Optional, List, Dict
import torch

MAX_CACHE_SIZE = 10000  # TODO: Make this configurable

class DataTuneLoader(torch.utils.data.DataLoader):
   """
   A data loader class for streaming data from Datatune views.
   Inherits from PyTorch's DataLoader class.

   Args:
       view (View): The view object to stream data from
       start_index (int): The starting index for streaming. Defaults to 0
       end_index (Optional[int]): The ending index for streaming. Defaults to None (stream until end)
       batch_size (Optional[int]): Number of samples per batch. Defaults to 32
       columns (Optional[List[str]]): List of column names to fetch. Defaults to None (fetch all columns)
       num_workers (Optional[int]): Number of worker threads. Defaults to 1

   Attributes:
       cache (List): Internal cache for storing pre-fetched batches
       cache_size (int): Current size of cached data
       thread (threading.Thread): Background thread for data fetching
       _view_size (int): Total number of rows in the view
   """

   def __init__(
       self,
       view,
       start_index: int = 0,
       end_index: Optional[int] = None,
       batch_size: Optional[int] = 32,
       columns: Optional[List[str]] = None,
       num_workers: Optional[int] = 1,
   ):
       self.view = view
       self.start_index = start_index
       self.end_index = end_index
       self.batch_size = batch_size
       self.columns = columns
       self.cache = []
       self.cache_size = 0
       self.thread = threading.Thread(target=self.bg_thread)
       self.num_workers = 1 if num_workers is None or num_workers < 1 else num_workers
       self._view_size = self._get_view_size()
       super(DataTuneLoader, self).__init__(
           self, batch_size=batch_size, shuffle=False, num_workers=num_workers
       )

   def _get_view_size(self) -> int:
       """
       Gets the total number of rows in the view.

       Returns:
           int: Total number of rows in the view
       """
       view_details = self.view.workspace.api.get_dataset_view_details(
           workspace_id=self.view.workspace.id,
           view_id=self.view.id
       )
       return view_details['total_rows']

   def _get_batch(self, start_index: int, end_index: int) -> Dict:
       """
       Fetches a batch of data from the view.

       Args:
           start_index (int): Starting index of the batch
           end_index (int): Ending index of the batch

       Returns:
           Dict: Dictionary containing the batch data and number of samples
       """
       batch_size = end_index - start_index
       batches = self.view.workspace.api.get_batches(
           workspace_id=self.view.workspace.id,
           view_id=self.view.id,
           start_index=start_index,
           batch_size=batch_size,
           num_batches=1
       )

       # Get the first (and only) batch from the generator
       batch_data = next(batches)
       
       # Restructure the data to match the required format
       final_batch = {col: [item[col] for item in batch_data] for col in self.columns}
       return {
           "data": final_batch,
           "num_samples": len(batch_data)
           }

   def bg_thread(self):
       """
       Background thread function for pre-fetching batches.
       Continuously fetches batches and adds them to the cache until the end index is reached.
       """
       curr_index = self.start_index
       end_index = self.end_index or self._view_size
       while curr_index < end_index:
           batch_end_index = min(curr_index + self.batch_size, end_index)
           try:
               batch = self._get_batch(curr_index, batch_end_index)
           except Exception as e:
               print(f"Error fetching batch: {e}")  # TODO: Add retry logic
               self.cache.append(e)
               break
           self.cache.append(batch)
           self.cache_size += batch["num_samples"]
           curr_index = batch_end_index
           while self.cache_size > MAX_CACHE_SIZE:
               time.sleep(1)
       self.cache.append(None)  # Signal end of data

   def reset(self):
       """
       Resets the data loader by clearing the cache and stopping the background thread.
       """
       if self.thread.is_alive():
           self.thread.join()
       self.cache = []
       self.cache_size = 0

   def stream(self):
       """
       Yields batches of data from the view.

       Yields:
           Dict: Dictionary containing the column data for each batch

       Raises:
           Exception: If there's an error fetching a batch
           AssertionError: If a requested column is not found in the batch
       """
       if not self.thread.is_alive():
           self.thread.start()
       while True:
           if not self.cache:
               time.sleep(0.1)
               continue
           batch = self.cache.pop(0)
           if batch is None:
               break
           if isinstance(batch, Exception):
               raise batch
           data = batch["data"]
           for col in self.columns:
               assert col in data, f"Column {col} not found in batch"
           yield data

   def __iter__(self):
       """Returns the stream iterator."""
       return self.stream()

   def __len__(self):
       """
       Returns the total number of batches.

       Returns:
           int: Number of batches in the dataset
       """
       return math.ceil(
           (self.end_index or self._view_size - self.start_index) / self.batch_size
       )

   def shutdown(self):
       """
       Cleanly shuts down the data loader by stopping the background thread.
       """
       if self.thread.is_alive():
           self.thread.join()