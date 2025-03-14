from typing import Iterable, Optional, Tuple, Union
from functools import partial


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
        if isinstance(item, tuple):
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


def slice_length(s: slice, length: int) -> int:
    s = s.indices(length)
    start, stop, step = s
    if step < 0:
        start, stop = stop, start
        step = -step
    return (stop - start) // step


def apply_int_on_slice(
    i: int, s: slice, length: int, base_length: Optional[int] = None
) -> int:
    start, stop, step = s
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
            i = 0
    resolved = start + i * step
    if resolved < start:
        raise IndexError(
            f"Index {i} is out of bounds for slice {s} over length {length}"
        )
    if resolved >= stop:
        raise IndexError(
            f"Index {i} is out of bounds for slice {s} over length {length}"
        )
    return resolved


def apply_int_on_iterable(i: int, indices: Iterable[int]) -> int:
    return list(indices)[i]


def apply_iterable_on_slice(
    indices: Iterable[int], s: slice, length: int
) -> Iterable[int]:
    start, stop, step = s
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
    if s1 is None:
        start = start2
    elif s2 is None:
        if s1 is not None:
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
        stop = start2 + stop1 * step2
    else:
        stop = min(stop2, start2 + stop1 * step2)
    return slice(start, stop, step)


def apply_slice(s1: INDEX_TYPE, s2: INDEX_TYPE, length: int) -> Union[
    slice,
    Iterable[int],
]:
    if s1 == slice(None):
        return s2
    if s2 == slice(None):
        return s1
    if isinstance(s1, slice) and isinstance(s2, slice):
        return apply_slice_on_slice(s1, s2)
    elif isinstance(s1, Iterable) and isinstance(s2, slice):
        return apply_iterable_on_slice(s1, s2, length)
    elif isinstance(s1, slice) and isinstance(s2, Iterable):
        return list(s2)[s1]
    elif isinstance(s1, Iterable) and isinstance(s2, Iterable):
        indices = list(s2)
        return [indices[i] for i in s1]
    raise ValueError("Invalid input types")
