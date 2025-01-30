import threading
import time
import math
from typing import Optional, List, Dict, Any, Generator, Union
import torch

MAX_CACHE_SIZE = 10000

class DataTuneLoader(torch.utils.data.DataLoader):
    """
    A data loader class for streaming data from Datatune views.
    Handles both regular view data and extra columns separately.

    Args:
        view (View): The view object to stream data from
        start_index (int): The starting index for streaming
        end_index (Optional[int]): The ending index for streaming
        batch_size (Optional[int]): Number of samples per batch
        columns (Optional[List[str]]): List of column names to fetch
        num_workers (Optional[int]): Number of worker threads
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
        self.batch_size = batch_size or 32
        self.columns = columns or []
        self.cache: List[Union[Dict, Exception, None]] = []
        self.cache_size = 0
        self.thread = threading.Thread(target=self.bg_thread)
        self.num_workers = max(1, num_workers or 1)

        # Initialize details
        self.view_details = None
        self.extra_columns_details = None
        self._initialize_details()
        
        # Set end index based on the maximum rows from both sources
        total_rows = max(
            self.view_details.get('total_rows', 0),
            max((col.get('num_rows', 0) for col in self.extra_columns_details.get('columns', [])), default=0)
        )
        self.end_index = end_index if end_index is not None else total_rows
        
        super().__init__(
            self, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

    def _initialize_details(self):
        """
        Initialize both view details and extra columns details.
        """
        try:
            # Get view details
            self.view_details = self.view.workspace.api.get_dataset_view_details(
                workspace_id=self.view.workspace.id,
                view_id=self.view.id
            )
            
            # Get extra columns details
            self.extra_columns_details = self.view.workspace.api.get_extra_columns_details(
                workspace_id=self.view.workspace.id,
                view_id=self.view.id
            )

            # Validate and filter columns
            available_columns = set()
            
            # Add columns from view details
            if self.view_details.get('slices'):
                for slice_item in self.view_details['slices']:
                    if 'column_maps' in slice_item:
                        available_columns.update(slice_item['column_maps'].values())
            
            # Add extra columns
            if self.extra_columns_details.get('columns'):
                available_columns.update(col['name'] for col in self.extra_columns_details['columns'])
            
            # Filter requested columns
            if self.columns:
                invalid_columns = [col for col in self.columns if col not in available_columns]
                if invalid_columns:
                    raise ValueError(f"Invalid columns requested: {invalid_columns}")
            else:
                self.columns = list(available_columns)

        except Exception as e:
            raise Exception(f"Failed to initialize view details: {str(e)}")

    def _get_batch(self, start_index: int, end_index: int) -> Dict[str, Any]:
        """
        Fetches a batch of data from the view, combining regular view data and extra columns.
        """
        batch_size = end_index - start_index
        try:
            # Get regular view data
            view_batch_data = []
            batches = self.view.workspace.api.get_batches(
                workspace_id=self.view.workspace.id,
                view_id=self.view.id,
                start_index=start_index,
                batch_size=batch_size,
                num_batches=1
            )
            
            for batch in batches:
                view_batch_data.extend(batch)

            # Initialize final batch with requested columns
            final_batch = {col: [] for col in self.columns}

            # Process view data if available
            if view_batch_data:
                for item in view_batch_data:
                    for col in self.columns:
                        final_batch[col].append(item.get(col))

            # Get extra column data if needed
            extra_column_names = [
                col['name'] for col in self.extra_columns_details.get('columns', [])
                if col['name'] in self.columns
            ]
            
            if extra_column_names and not view_batch_data:  # If no view data but we have extra columns
                # Create empty rows for the batch size
                for col in extra_column_names:
                    final_batch[col] = [None] * batch_size

                # Fill in actual extra column data
                for col_name in extra_column_names:
                    column_data = self.view.workspace.api.get_column_data(
                        workspace_id=self.view.workspace.id,
                        view_id=self.view.id,
                        column_name=col_name,
                        start_index=start_index,
                        batch_size=batch_size
                    )
                    for i, value in enumerate(column_data):
                        if i < batch_size:
                            final_batch[col_name][i] = value

            return {
                "data": final_batch,
                "num_samples": len(view_batch_data) or batch_size
            }
            
        except Exception as e:
            raise Exception(f"Error fetching batch: {str(e)}")

    def bg_thread(self):
        """
        Background thread for pre-fetching batches.
        """
        curr_index = self.start_index
        consecutive_empty = 0
        
        while curr_index < self.end_index and consecutive_empty < 3:
            batch_end_index = min(curr_index + self.batch_size, self.end_index)
            
            try:
                batch = self._get_batch(curr_index, batch_end_index)
                if batch["num_samples"] == 0:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0
                    self.cache.append(batch)
                    self.cache_size += batch["num_samples"]
                    curr_index = batch_end_index
                
                while self.cache_size > MAX_CACHE_SIZE:
                    time.sleep(0.1)
                    
            except Exception as e:
                self.cache.append(e)
                break

        self.cache.append(None)

    def reset(self):
        """Resets the data loader."""
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        self.cache = []
        self.cache_size = 0
        self._initialize_details()  # Re-initialize details
        self.thread = threading.Thread(target=self.bg_thread)

    def stream(self) -> Generator[Dict[str, List], None, None]:
        """
        Streams batches of data from the view.
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
                
            yield batch["data"]

    def __iter__(self):
        return self.stream()

    def __len__(self) -> int:
        """Returns the total number of batches."""
        total_items = self.end_index - self.start_index
        return max(1, math.ceil(total_items / self.batch_size))

    def shutdown(self):
        """Cleanly shuts down the data loader."""
        if self.thread.is_alive():
            self.thread.join(timeout=5)