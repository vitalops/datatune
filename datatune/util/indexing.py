from typing import Iterable, Optional, Tuple, Union, List
from functools import partial
from math import ceil


ROW_INDEX_TYPE = Union[int, slice, Iterable[int]]
COLUMN_INDEX_TYPE = Union[str, Iterable[str]]
INDEX_TYPE = Union[
    ROW_INDEX_TYPE, COLUMN_INDEX_TYPE, Tuple[ROW_INDEX_TYPE, COLUMN_INDEX_TYPE]
]


def parse_row_and_column_indices(
    item: INDEX_TYPE,
) -> Tuple[ROW_INDEX_TYPE, COLUMN_INDEX_TYPE]:
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
    return list(indices)[i]


def apply_iterable_on_slice(
    indices: Iterable[int], s: slice, length: int
) -> Iterable[int]:
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
    indices2 = list(indices2)
    return [indices2[i] for i in indices]


def apply_slice(
    s1: ROW_INDEX_TYPE, s2: ROW_INDEX_TYPE, length: int
) -> Union[slice, Iterable[int],]:
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


def get_row_groups_for_slice(base_length, row_group_sizes, row_group_offsets, sl: ROW_INDEX_TYPE) -> List[int]:
    if isinstance(sl, int):
        # Convert single index to iterable
        adjusted_idx = sl if sl >= 0 else sl + base_length

        # Ensure index is in bounds
        if not (0 <= adjusted_idx < base_length):
            # This ensures an IndexError will be raised later
            return []

        return get_row_groups_for_slice(base_length, row_group_sizes, row_group_offsets, [adjusted_idx])

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
        indices = sorted(
            set(int(i) if i >= 0 else int(i + base_length) for i in sl)
        )
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
    base_length, row_group_sizes, row_group_offsets, sl: ROW_INDEX_TYPE, group_idx: int
) -> ROW_INDEX_TYPE:
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