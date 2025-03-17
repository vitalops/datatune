from functools import partial
from math import ceil
from typing import Iterable, List, Optional, Tuple, Union

ROW_INDEX_TYPE = Union[int, slice, Iterable[int]]
COLUMN_INDEX_TYPE = Union[str, Iterable[str]]
INDEX_TYPE = Union[
    ROW_INDEX_TYPE, COLUMN_INDEX_TYPE, Tuple[ROW_INDEX_TYPE, COLUMN_INDEX_TYPE]
]


def parse_row_and_column_indices(
    item: INDEX_TYPE,
) -> Tuple[ROW_INDEX_TYPE, COLUMN_INDEX_TYPE]:
    """
    Parse an index item into row and column components.

    This function handles various input formats used to index a dataset:
    - String: Interpreted as a column name (column index with all rows)
    - Integer: Interpreted as a row index (one row with all columns)
    - Slice: Interpreted as a row slice (multiple rows with all columns)
    - Iterable of strings: Interpreted as multiple column names
    - Iterable of integers: Interpreted as multiple row indices
    - Tuple of (column, row): Interpreted as specific column(s) and row(s)

    Parameters:
    -----------
    item : INDEX_TYPE
        The index item to parse

    Returns:
    --------
    Tuple[ROW_INDEX_TYPE, COLUMN_INDEX_TYPE]
        Row and column indices
    """
    if isinstance(item, str):
        return slice(None), item
    if isinstance(item, int):
        return item, None
    if isinstance(item, slice):
        return item, None
    if isinstance(item, Iterable):
        if not isinstance(item, (list, tuple)):
            item = tuple(item)
        if not item:
            return slice(None), []
        if isinstance(item[0], str):
            if len(item) == 1:
                return slice(None), item[0]
            if isinstance(item[1], str):
                return slice(None), item
            else:
                assert len(item) == 2
                return item[1], item[0]
        return item, None


def slice_length(s: ROW_INDEX_TYPE, length: int) -> int:
    """
    Calculate the number of elements that a slice or index will produce.

    This function handles various index types:
    - Integer: Single element (length 1)
    - Slice: Number of elements in the slice applied to a sequence of the given length
    - Iterable: Number of elements in the iterable

    Parameters:
    -----------
    s : ROW_INDEX_TYPE
        The slice, integer index, or iterable of indices
    length : int
        The length of the sequence being indexed

    Returns:
    --------
    int
        The number of elements that would be selected by the index
    """
    if isinstance(s, int):
        return 1
    if hasattr(s, "__len__"):
        return len(s)
    start, stop, step = s.start, s.stop, s.step
    if start is None:
        start = 0
    elif start < 0:
        start += length
        if start < 0:
            start = 0
    if step is None:
        step = 1
    elif step < 0:
        start, stop = stop, start
        step = -step
    if stop is None or stop > length:
        stop = length
    elif stop < 0:
        stop += length
        if stop < 0:
            stop = 0
    ret = ceil((stop - start) / step)
    return 0 if ret < 0 else ret


def apply_int_on_slice(
    i: int, s: slice, length: int, base_length: Optional[int] = None
) -> int:
    """
    Apply an integer index to a sliced sequence to get the final index.

    This function converts an index into a sliced sequence to the corresponding
    index in the original sequence.

    Parameters:
    -----------
    i : int
        The index within the sliced sequence
    s : slice
        The slice applied to the original sequence
    length : int
        The length of the original sequence
    base_length : Optional[int]
        The length of the sliced sequence, calculated if not provided

    Returns:
    --------
    int
        The index in the original sequence

    Raises:
    -------
    IndexError
        If the index is out of bounds
    """
    if length <= 0:
        raise IndexError("Index is out of bounds for empty slice")
    orig_i = i
    orig_s = s
    start, stop, step = s.start, s.stop, s.step
    if start is None:
        start = 0
    elif start < 0:
        start += length
        if start < 0:
            start = 0
    if step is None:
        step = 1
    if stop is None:
        stop = length
    elif stop < 0:
        stop += length
        if stop < 0:
            stop = 0
    if i < 0:
        if base_length is None:
            base_length = slice_length(s, length)
        i += base_length
        if i < 0:
            raise IndexError(
                f"Index {orig_i} is out of bounds for slice {orig_s} over length {length}"
            )
    resolved_index = start + i * step
    if (step > 0 and ((resolved_index < start) or (resolved_index >= stop))) or (
        step < 0 and ((resolved_index > start) or (resolved_index <= stop))
    ):
        raise IndexError(
            f"Index {orig_i} is out of bounds for slice {orig_s} over length {length}"
        )
    return resolved_index


def apply_int_on_iterable(i: int, indices: Iterable[int]) -> int:
    """
    Apply an integer index to an iterable of indices to get the final index.

    Parameters:
    -----------
    i : int
        The index within the iterable
    indices : Iterable[int]
        The iterable of indices

    Returns:
    --------
    int
        The index at the specified position in the iterable
    """
    return list(indices)[i]


def apply_iterable_on_slice(
    indices: Iterable[int], s: slice, length: int
) -> Iterable[int]:
    """
    Apply an iterable of indices to a sliced sequence to get the final indices.

    Parameters:
    -----------
    indices : Iterable[int]
        The indices within the sliced sequence
    s : slice
        The slice applied to the original sequence
    length : int
        The length of the original sequence

    Returns:
    --------
    Iterable[int]
        The corresponding indices in the original sequence
    """
    start, stop, step = s.start, s.stop, s.step
    if start is None:
        start = 0
    if start < 0:
        start += length
        if start < 0:
            start = 0
    if step is None:
        step = 1
    if stop is None:
        stop = length
    if stop < 0:
        stop += length
        if stop < 0:
            stop = 0
    base_length = slice_length(s, length)
    return list(
        map(
            partial(apply_int_on_slice, s=s, length=length, base_length=base_length),
            indices,
        )
    )


def apply_slice_on_slice(s1: slice, s2: slice) -> slice:
    """
    Compose two slices to create a new slice that has the same effect as applying both sequentially.

    Parameters:
    -----------
    s1 : slice
        The first slice to apply
    s2 : slice
        The second slice to apply

    Returns:
    --------
    slice
        A new slice that has the same effect as applying s1 and then s2
    """
    if s1 == slice(None):
        return s2
    if s2 == slice(None):
        return s1
    start = None
    stop = None
    step = None
    start1 = s1.start
    stop1 = s1.stop
    step1 = s1.step
    start2 = s2.start
    stop2 = s2.stop
    step2 = s2.step
    if start1 is None:
        start = start2
    elif start2 is None:
        if start1 is not None:
            start = start1
    else:
        start = start2 + start1 * step2
    if step1 is None:
        step = step2
    elif step2 is None:
        step = step1
    else:
        step = step1 * step2
    if stop1 is None:
        stop = stop2
    elif stop2 is None:
        stop = (start2 or 0) + stop1 * step2
    else:
        stop = min(stop2, (start2 or 0) + stop1 * step2)
    return slice(start, stop, step)


def apply_iterable_on_iterable(
    indices: Iterable[int], indices2: Iterable[int]
) -> Iterable[int]:
    """
    Apply an iterable of indices to another iterable to get the final indices.

    Parameters:
    -----------
    indices : Iterable[int]
        The indices to apply
    indices2 : Iterable[int]
        The iterable to index into

    Returns:
    --------
    Iterable[int]
        The final indices after applying both iterables
    """
    indices2 = list(indices2)
    return [indices2[i] for i in indices]


def apply_slice(
    s1: ROW_INDEX_TYPE, s2: ROW_INDEX_TYPE, length: int
) -> Union[slice, Iterable[int],]:
    """
    Apply one index to another to create a composite index.

    This function handles various combinations of index types:
    - slice + slice: Creates a composed slice
    - iterable + slice: Applies the iterable to the slice
    - int + slice: Gets a specific element from the slice
    - slice + iterable: Slices the iterable
    - iterable + iterable: Applies one iterable to another
    - int + iterable: Gets a specific element from the iterable

    Parameters:
    -----------
    s1 : ROW_INDEX_TYPE
        The first index to apply
    s2 : ROW_INDEX_TYPE
        The second index to apply
    length : int
        The length of the sequence being indexed

    Returns:
    --------
    Union[slice, Iterable[int]]
        The composite index

    Raises:
    -------
    ValueError
        If the second index is an integer
    TypeError
        If the index types are invalid
    """
    if s1 == slice(None):
        return s2
    if s2 == slice(None):
        return s1
    if isinstance(s1, slice) and isinstance(s2, slice):
        return apply_slice_on_slice(s1, s2)
    elif isinstance(s1, Iterable) and isinstance(s2, slice):
        return apply_iterable_on_slice(s1, s2, length)
    elif isinstance(s1, int) and isinstance(s2, slice):
        return apply_int_on_slice(s1, s2, length)
    elif isinstance(s1, slice) and isinstance(s2, Iterable):
        return list(s2)[s1]
    elif isinstance(s1, Iterable) and isinstance(s2, Iterable):
        return apply_iterable_on_iterable(s1, s2)
    elif isinstance(s1, int) and isinstance(s2, Iterable):
        return apply_int_on_iterable(s1, s2)
    elif isinstance(s2, int):
        raise ValueError("Cannot index a scalar.")
    raise TypeError(f"Invalid index types {s1} and {s2}. Expected {ROW_INDEX_TYPE}.")


def get_row_groups_for_slice(
    base_length: int,
    row_group_sizes: List[int],
    row_group_offsets: List[int],
    sl: ROW_INDEX_TYPE,
) -> List[int]:
    """
    Determine which row groups contain the rows referenced by a slice or index.

    This function identifies which row groups in a partitioned dataset need to be
    read to satisfy a particular row selection.

    Parameters:
    -----------
    base_length : int
        The total number of rows in the dataset
    row_group_sizes : List[int]
        The size (number of rows) of each row group
    row_group_offsets : List[int]
        The starting offset of each row group
    sl : ROW_INDEX_TYPE
        The slice, integer index, or iterable of indices to evaluate

    Returns:
    --------
    List[int]
        A list of row group indices that contain the requested rows

    Raises:
    -------
    ValueError
        If the slice type is unsupported
    """
    if isinstance(sl, int):
        # Convert single index to iterable
        adjusted_idx = sl if sl >= 0 else sl + base_length

        # Ensure index is in bounds
        if not (0 <= adjusted_idx < base_length):
            # This ensures an IndexError will be raised later
            return []

        return get_row_groups_for_slice(
            base_length, row_group_sizes, row_group_offsets, [adjusted_idx]
        )

    elif isinstance(sl, slice):
        # Handle slice
        start, stop, step = sl.start, sl.stop, sl.step
        if start is None:
            start = 0
        if stop is None:
            stop = base_length
        if step is None:
            step = 1

        # Adjust negative indices
        if start < 0:
            start += base_length
        if stop < 0:
            stop += base_length

        # Ensure bounds
        start = max(0, min(start, base_length))
        stop = max(0, min(stop, base_length))

        # Find row groups that contain rows in the slice
        required_groups = []
        for i, offset in enumerate(row_group_offsets):
            group_end = offset + row_group_sizes[i]
            # If this row group contains any rows in the slice
            if (offset < stop) and (group_end > start):
                required_groups.append(i)
        return required_groups

    elif isinstance(sl, Iterable):
        # Handle list of indices
        indices = sorted(set(int(i) if i >= 0 else int(i + base_length) for i in sl))
        required_groups = set()

        for idx in indices:
            if 0 <= idx < base_length:
                # Find the row group containing this index
                for group_idx, offset in enumerate(row_group_offsets):
                    if idx < offset + row_group_sizes[group_idx]:
                        required_groups.add(group_idx)
                        break

        return sorted(required_groups)

    # Should not reach here due to type checking in the indexing utils
    raise ValueError(f"Unsupported index type: {type(sl)}")


def map_slice_to_row_group(
    base_length: int,
    row_group_sizes: List[int],
    row_group_offsets: List[int],
    sl: ROW_INDEX_TYPE,
    group_idx: int,
) -> ROW_INDEX_TYPE:
    """
    Map a global dataset slice to a local slice within a specific row group.

    This function transforms indices that reference the entire dataset into
    indices that reference rows within a specific row group.

    Parameters:
    -----------
    base_length : int
        The total number of rows in the dataset
    row_group_sizes : List[int]
        The size (number of rows) of each row group
    row_group_offsets : List[int]
        The starting offset of each row group
    sl : ROW_INDEX_TYPE
        The slice, integer index, or iterable of indices to map
    group_idx : int
        The index of the row group to map to

    Returns:
    --------
    ROW_INDEX_TYPE or None
        The mapped indices within the row group, or None if the slice doesn't
        overlap with this row group
    """
    group_offset = row_group_offsets[group_idx]
    group_size = row_group_sizes[group_idx]

    if isinstance(sl, int):
        # Single index - convert to local index
        adjusted_idx = sl if sl >= 0 else sl + base_length
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
            stop = base_length
        if step is None:
            step = 1

        # Adjust for negative indices
        if start < 0:
            start += base_length
        if stop < 0:
            stop += base_length

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
                idx += base_length
            local_idx = idx - group_offset
            if 0 <= local_idx < group_size:
                local_indices.append(local_idx)
        return local_indices if local_indices else None

    # Should not reach here
    return None
