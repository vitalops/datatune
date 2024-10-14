import threading
import time
import math
from typing import Optional, List, Dict
import torch

MAX_CACHE_SIZE = 10000  # TODO: Make this configurable

class DataTuneLoader(torch.utils.data.DataLoader):
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
        view_details = self.view.workspace.api.get_dataset_view_details(
            workspace_id=self.view.workspace.id,
            view_id=self.view.id
        )
        return view_details['total_rows']

    def _get_batch(self, start_index: int, end_index: int) -> Dict:
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
        if self.thread.is_alive():
            self.thread.join()
        self.cache = []
        self.cache_size = 0

    def stream(self):
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
        return self.stream()

    def __len__(self):
        return math.ceil(
            (self.end_index or self._view_size - self.start_index) / self.batch_size
        )

    def shutdown(self):
        if self.thread.is_alive():
            self.thread.join()