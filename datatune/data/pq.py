import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from typing import List, Dict, Union, Optional, Iterable, Any
from datatune.data.dataset import Dataset, Column
from datatune.util.indexing import ROW_INDEX_TYPE, slice_length
from copy import deepcopy


class ParquetDataset(Dataset):
    def __init__(self, parquet_path: str):
        super().__init__()
        self.parquet_path = parquet_path
        
        # Open the file for metadata only - doesn't load data
        self.parquet_file = pq.ParquetFile(parquet_path)
        
        # Get file metadata
        self.num_row_groups = self.parquet_file.num_row_groups
        self.base_length = self.parquet_file.metadata.num_rows
        
        # Store row group sizes for efficient slicing
        self.row_group_sizes = [self.parquet_file.metadata.row_group(i).num_rows 
                                for i in range(self.num_row_groups)]
        self.row_group_offsets = [0]
        for size in self.row_group_sizes[:-1]:
            self.row_group_offsets.append(self.row_group_offsets[-1] + size)

        
        # Get schema using PyArrow's schema property
        schema = self.parquet_file.schema_arrow
        
        for i, field in enumerate(schema.names):
            # Get the field type from the PyArrow schema
            arrow_type = schema.types[i]
            
            # Convert Arrow type to numpy dtype
            if pa.types.is_integer(arrow_type):
                np_type = np.dtype('int64')
            elif pa.types.is_floating(arrow_type):
                np_type = np.dtype('float64')
            elif pa.types.is_boolean(arrow_type):
                np_type = np.dtype('bool')
            elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
                np_type = np.dtype('object')
            elif pa.types.is_timestamp(arrow_type):
                np_type = np.dtype('datetime64[ns]')
            else:
                # Default to object for complex types
                np_type = np.dtype('object')
                
            self.columns[field] = Column(field, np_type)
    
    def copy(self) -> "ParquetDataset":
        # Create a new instance with the same parquet file
        new_ds = ParquetDataset(self.parquet_path)
        
        # Copy the columns dictionary (this can be safely deepcopied)
        new_ds.columns = deepcopy(self.columns)
        
        # Copy the slice
        new_ds.slice = deepcopy(self.slice)
        
        # The other attributes are recreated in the constructor
        return new_ds
    
    def _get_row_groups_for_slice(self, sl: ROW_INDEX_TYPE) -> List[int]:
        if isinstance(sl, int):
            # Convert single index to iterable
            adjusted_idx = sl if sl >= 0 else sl + self.base_length
            
            # Ensure index is in bounds
            if not (0 <= adjusted_idx < self.base_length):
                # This ensures an IndexError will be raised later
                return []
                
            return self._get_row_groups_for_slice([adjusted_idx])
            
        elif isinstance(sl, slice):
            # Handle slice
            start, stop, step = sl.start, sl.stop, sl.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.base_length
            if step is None:
                step = 1
                
            # Adjust negative indices
            if start < 0:
                start += self.base_length
            if stop < 0:
                stop += self.base_length
                
            # Ensure bounds
            start = max(0, min(start, self.base_length))
            stop = max(0, min(stop, self.base_length))
            
            # Find row groups that contain rows in the slice
            required_groups = []
            for i, offset in enumerate(self.row_group_offsets):
                group_end = offset + self.row_group_sizes[i]
                # If this row group contains any rows in the slice
                if (offset < stop) and (group_end > start):
                    required_groups.append(i)
            return required_groups
            
        elif isinstance(sl, Iterable):
            # Handle list of indices
            indices = sorted(set(int(i) if i >= 0 else int(i + self.base_length) for i in sl))
            required_groups = set()
            
            for idx in indices:
                if 0 <= idx < self.base_length:
                    # Find the row group containing this index
                    for group_idx, offset in enumerate(self.row_group_offsets):
                        if idx < offset + self.row_group_sizes[group_idx]:
                            required_groups.add(group_idx)
                            break
                            
            return sorted(required_groups)
            
        # Should not reach here due to type checking in the indexing utils
        raise ValueError(f"Unsupported index type: {type(sl)}")
    
    def _map_slice_to_row_group(self, sl: ROW_INDEX_TYPE, group_idx: int) -> ROW_INDEX_TYPE:
        group_offset = self.row_group_offsets[group_idx]
        group_size = self.row_group_sizes[group_idx]
        
        if isinstance(sl, int):
            # Single index - convert to local index
            adjusted_idx = sl if sl >= 0 else sl + self.base_length
            local_idx = adjusted_idx - group_offset
            
            # Check if the index is in this group
            if 0 <= local_idx < group_size:
                return [local_idx]  # Return as iterable
            return []  # Index not in this group, return empty list
            
        elif isinstance(sl, slice):
            # Slice
            start, stop, step = sl.start, sl.stop, sl.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.base_length
            if step is None:
                step = 1
                
            # Adjust for negative indices
            if start < 0:
                start += self.base_length
            if stop < 0:
                stop += self.base_length
                
            # Map to local coordinates
            local_start = max(0, start - group_offset)
            local_stop = min(group_size, stop - group_offset)
            
            if local_start >= group_size or local_stop <= 0:
                return None  # Slice doesn't overlap this group
                
            return slice(local_start, local_stop, step)
            
        elif isinstance(sl, Iterable):
            # List of indices
            local_indices = []
            for idx in sl:
                if idx < 0:
                    idx += self.base_length
                local_idx = idx - group_offset
                if 0 <= local_idx < group_size:
                    local_indices.append(local_idx)
            return local_indices if local_indices else None
            
        # Should not reach here
        return None
    
    def realize(self):
        import pandas as pd
        
        # Determine which row groups contain our slice
        row_groups = self._get_row_groups_for_slice(self.slice)
        
        # Get column names to read
        column_names = list(self.columns.keys())
        
        # Special case for integer index (single row)
        if isinstance(self.slice, int):
            # Convert to absolute index
            target_row = self.slice if self.slice >= 0 else self.slice + self.base_length
            
            # Check bounds
            if not (0 <= target_row < self.base_length):
                raise IndexError(f"Index {self.slice} is out of bounds for dataset with length {self.base_length}")
            
            # Find which row group contains this row
            target_group = None
            local_idx = None
            
            for i, offset in enumerate(self.row_group_offsets):
                if target_row < offset + self.row_group_sizes[i]:
                    target_group = i
                    local_idx = target_row - offset
                    break
                    
            if target_group is not None:
                # Read just the specific row from the appropriate row group
                table = self.parquet_file.read_row_group(target_group, columns=column_names)
                df = table.slice(local_idx, 1).to_pandas()
                return df
            else:
                # This should not happen if bounds check passed
                raise IndexError(f"Index {self.slice} could not be located in any row group")
        
        # For other slice types, we need to read multiple rows
        result_tables = []
        
        # Handle empty row_groups (out of bounds indices)
        if not row_groups and isinstance(self.slice, Iterable):
            raise IndexError(f"Indices {self.slice} are out of bounds for dataset with length {self.base_length}")
        
        for group_idx in row_groups:
            # Map the slice to this row group's local coordinates
            local_slice = self._map_slice_to_row_group(self.slice, group_idx)
            
            if local_slice is not None:
                # Read the row group
                table = self.parquet_file.read_row_group(group_idx, columns=column_names)
                
                # Apply the local slice to the row group
                if isinstance(local_slice, list):
                    if len(local_slice) == 1:
                        # Single index
                        sliced_table = table.slice(local_slice[0], 1)
                    else:
                        # For lists of indices, we need to read individual rows
                        rows = [table.slice(idx, 1) for idx in local_slice]
                        sliced_table = pa.concat_tables(rows) if rows else None
                elif isinstance(local_slice, slice):
                    start = local_slice.start or 0
                    length = slice_length(local_slice, self.row_group_sizes[group_idx])
                    sliced_table = table.slice(start, length)
                else:
                    # This should not happen with the modified code
                    raise ValueError(f"Unexpected local_slice type: {type(local_slice)}")
                
                if sliced_table is not None:
                    result_tables.append(sliced_table)
        
        # Combine all tables and convert to pandas
        if result_tables:
            final_table = pa.concat_tables(result_tables)
            return final_table.to_pandas()
        else:
            # Empty result could mean out of bounds
            if isinstance(self.slice, (int, Iterable)):
                raise IndexError(f"Index {self.slice} is out of bounds for dataset with length {self.base_length}")
            # Return empty DataFrame with correct columns for empty slices
            return pd.DataFrame(columns=column_names)
            
    def get_statistics(self) -> Dict:
        stats = {
            "num_rows": self.base_length,
            "num_row_groups": self.num_row_groups,
            "columns": {},
            "file_size_bytes": self.parquet_file.metadata.serialized_size,
        }
        
        # Get column-level statistics if available
        schema = self.parquet_file.schema_arrow
        for col_name in self.columns:
            col_stats = {"min_values": [], "max_values": [], "null_counts": []}
            
            # Find column index in schema
            try:
                col_idx = schema.names.index(col_name)
                
                for i in range(self.num_row_groups):
                    row_group = self.parquet_file.metadata.row_group(i)
                    col_chunk = row_group.column(col_idx)
                    
                    if col_chunk.statistics:
                        stats_obj = col_chunk.statistics
                        if stats_obj.has_min_max:
                            col_stats["min_values"].append(stats_obj.min)
                            col_stats["max_values"].append(stats_obj.max)
                        col_stats["null_counts"].append(stats_obj.null_count)
            except (KeyError, ValueError):
                # Column not found or no statistics
                pass
                
            stats["columns"][col_name] = col_stats
            
        return stats
        
    def head(self, n: int = 5):
        # Create a new dataset with the first n rows only
        temp_dataset = self.copy()
        temp_dataset.slice = slice(0, n)
        return temp_dataset.realize()