import pytest


from datatune.util.indexing import (
    slice_length,
    apply_int_on_slice,
    apply_int_on_iterable,
    apply_iterable_on_slice,
    apply_slice_on_slice,
    apply_slice,
)

#######################
# Tests for slice_length
#######################


def test_slice_length_basic():
    """Test basic functionality of slice_length."""
    assert slice_length(slice(0, 10, 1), 10) == 10
    assert slice_length(slice(0, 5, 1), 10) == 5
    assert slice_length(slice(5, 10, 1), 10) == 5


def test_slice_length_with_step():
    """Test slice_length with different step values."""
    assert slice_length(slice(0, 10, 2), 10) == 5
    assert slice_length(slice(0, 10, 3), 10) == 4
    assert slice_length(slice(0, 10, 5), 10) == 2


def test_slice_length_negative_step():
    """Test slice_length with negative step values."""
    assert slice_length(slice(10, 0, -1), 10) == 10
    assert slice_length(slice(10, 5, -1), 10) == 5
    assert slice_length(slice(5, 0, -1), 10) == 5


def test_slice_length_with_none():
    """Test slice_length when slice contains None values."""
    assert slice_length(slice(None, None, None), 10) == 10
    assert slice_length(slice(None, 5, None), 10) == 5
    assert slice_length(slice(5, None, None), 10) == 5
    assert slice_length(slice(None, None, 2), 10) == 5


def test_slice_length_negative_indices():
    """Test slice_length with negative indices."""
    assert slice_length(slice(-10, None, None), 10) == 10
    assert slice_length(slice(-5, None, None), 10) == 5
    assert slice_length(slice(None, -5, None), 10) == 5
    assert slice_length(slice(-8, -3, None), 10) == 5


def test_slice_length_out_of_bounds():
    """Test slice_length with indices that are out of bounds."""
    assert slice_length(slice(0, 20, 1), 10) == 10
    assert slice_length(slice(-20, 20, 1), 10) == 10
    assert slice_length(slice(-20, -10, 1), 10) == 0


def test_slice_length_empty_slice():
    """Test slice_length with slices that produce empty results."""
    assert slice_length(slice(5, 5, 1), 10) == 0
    assert slice_length(slice(10, 0, 1), 10) == 0
    assert slice_length(slice(0, 10, -1), 10) == 0


#######################
# Tests for apply_int_on_slice
#######################


def test_apply_int_on_slice_basic():
    """Test basic functionality of apply_int_on_slice."""
    assert apply_int_on_slice(0, slice(0, 10, 1), 10) == 0
    assert apply_int_on_slice(5, slice(0, 10, 1), 10) == 5
    assert apply_int_on_slice(0, slice(5, 10, 1), 10) == 5
    assert apply_int_on_slice(2, slice(5, 10, 1), 10) == 7


def test_apply_int_on_slice_with_step():
    """Test apply_int_on_slice with different step values."""
    assert apply_int_on_slice(0, slice(0, 10, 2), 10) == 0
    assert apply_int_on_slice(1, slice(0, 10, 2), 10) == 2
    assert apply_int_on_slice(4, slice(0, 10, 2), 10) == 8
    assert apply_int_on_slice(0, slice(1, 10, 2), 10) == 1
    assert apply_int_on_slice(1, slice(1, 10, 2), 10) == 3


def test_apply_int_on_slice_negative_step():
    """Test apply_int_on_slice with negative step values."""
    assert apply_int_on_slice(0, slice(9, 0, -1), 10) == 9
    assert apply_int_on_slice(1, slice(9, 0, -1), 10) == 8
    assert apply_int_on_slice(8, slice(9, 0, -1), 10) == 1


def test_apply_int_on_slice_with_none():
    """Test apply_int_on_slice when slice contains None values."""
    assert apply_int_on_slice(0, slice(None, None, None), 10) == 0
    assert apply_int_on_slice(5, slice(None, None, None), 10) == 5
    assert apply_int_on_slice(0, slice(5, None, None), 10) == 5
    assert apply_int_on_slice(0, slice(None, 5, None), 10) == 0
    assert apply_int_on_slice(0, slice(None, None, 2), 10) == 0
    assert apply_int_on_slice(1, slice(None, None, 2), 10) == 2


def test_apply_int_on_slice_negative_indices():
    """Test apply_int_on_slice with negative indices."""
    assert apply_int_on_slice(0, slice(-5, None, None), 10) == 5
    assert apply_int_on_slice(0, slice(None, -5, None), 10) == 0
    assert apply_int_on_slice(0, slice(-8, -3, None), 10) == 2
    assert apply_int_on_slice(0, slice(-1, -10, -1), 10) == 9


def test_apply_int_on_slice_negative_i():
    """Test apply_int_on_slice with negative i values."""
    assert apply_int_on_slice(-1, slice(0, 10, 1), 10) == 9
    assert apply_int_on_slice(-2, slice(0, 10, 1), 10) == 8
    assert apply_int_on_slice(-1, slice(0, 10, 2), 10, 5) == 8
    assert apply_int_on_slice(-1, slice(9, 0, -1), 10) == 1


def test_apply_int_on_slice_out_of_bounds_i():
    """Test apply_int_on_slice with i values that are out of bounds."""
    with pytest.raises(IndexError):
        apply_int_on_slice(10, slice(0, 10, 1), 10)

    with pytest.raises(IndexError):
        apply_int_on_slice(5, slice(0, 5, 1), 10)

    with pytest.raises(IndexError):
        apply_int_on_slice(-11, slice(0, 10, 1), 10)


#######################
# Tests for apply_int_on_iterable
#######################


def test_apply_int_on_iterable_basic():
    """Test basic functionality of apply_int_on_iterable."""
    assert apply_int_on_iterable(0, [1, 2, 3, 4, 5]) == 1
    assert apply_int_on_iterable(2, [1, 2, 3, 4, 5]) == 3
    assert apply_int_on_iterable(4, [1, 2, 3, 4, 5]) == 5


def test_apply_int_on_iterable_negative_index():
    """Test apply_int_on_iterable with negative indices."""
    assert apply_int_on_iterable(-1, [1, 2, 3, 4, 5]) == 5
    assert apply_int_on_iterable(-3, [1, 2, 3, 4, 5]) == 3
    assert apply_int_on_iterable(-5, [1, 2, 3, 4, 5]) == 1


def test_apply_int_on_iterable_empty():
    """Test apply_int_on_iterable with an empty iterable."""
    with pytest.raises(IndexError):
        apply_int_on_iterable(0, [])


def test_apply_int_on_iterable_out_of_bounds():
    """Test apply_int_on_iterable with indices that are out of bounds."""
    with pytest.raises(IndexError):
        apply_int_on_iterable(5, [1, 2, 3, 4, 5])

    with pytest.raises(IndexError):
        apply_int_on_iterable(-6, [1, 2, 3, 4, 5])


def test_apply_int_on_iterable_generator():
    """Test apply_int_on_iterable with a generator."""
    gen = (i for i in range(1, 6))
    assert apply_int_on_iterable(0, gen) == 1
    # Note: generator is now consumed
    gen = (i for i in range(1, 6))
    assert apply_int_on_iterable(4, gen) == 5


#######################
# Tests for apply_iterable_on_slice
#######################


def test_apply_iterable_on_slice_basic():
    """Test basic functionality of apply_iterable_on_slice."""
    assert apply_iterable_on_slice([0, 1, 2], slice(0, 10, 1), 10) == [0, 1, 2]
    assert apply_iterable_on_slice([0, 1, 2], slice(5, 10, 1), 10) == [5, 6, 7]
    assert apply_iterable_on_slice([0, 1, 2], slice(0, 10, 2), 10) == [0, 2, 4]


def test_apply_iterable_on_slice_negative_indices():
    """Test apply_iterable_on_slice with negative indices in the iterable."""
    assert apply_iterable_on_slice([-1, -2, -3], slice(0, 10, 1), 10) == [9, 8, 7]
    assert apply_iterable_on_slice([-1, -2, -3], slice(5, 10, 1), 10) == [9, 8, 7]
    assert apply_iterable_on_slice([-1, -2, -3], slice(0, 10, 2), 10) == [8, 6, 4]


def test_apply_iterable_on_slice_with_none():
    """Test apply_iterable_on_slice when slice contains None values."""
    assert apply_iterable_on_slice([0, 1, 2], slice(None, None, None), 10) == [0, 1, 2]
    assert apply_iterable_on_slice([0, 1, 2], slice(5, None, None), 10) == [5, 6, 7]
    assert apply_iterable_on_slice([0, 1, 2], slice(None, 5, None), 10) == [0, 1, 2]
    assert apply_iterable_on_slice([0, 1, 2], slice(None, None, 2), 10) == [0, 2, 4]


def test_apply_iterable_on_slice_out_of_bounds():
    """Test apply_iterable_on_slice with indices in the iterable that are out of bounds."""
    with pytest.raises(IndexError):
        apply_iterable_on_slice([5, 6, 7], slice(0, 5, 1), 10)

    with pytest.raises(IndexError):
        apply_iterable_on_slice([0, 1, 2, 3, 4, 5], slice(0, 5, 1), 10)

    with pytest.raises(IndexError):
        apply_iterable_on_slice([-6, -7, -8], slice(0, 5, 1), 10)


def test_apply_iterable_on_slice_empty_iterable():
    """Test apply_iterable_on_slice with an empty iterable."""
    assert apply_iterable_on_slice([], slice(0, 10, 1), 10) == []


def test_apply_iterable_on_slice_negative_step():
    """Test apply_iterable_on_slice with negative step values."""
    assert apply_iterable_on_slice([0, 1, 2], slice(9, 0, -1), 10) == [9, 8, 7]
    assert apply_iterable_on_slice([-1, -2, -3], slice(9, 0, -1), 10) == [1, 2, 3]


def test_apply_iterable_on_slice_generator():
    """Test apply_iterable_on_slice with a generator."""
    gen = (i for i in range(3))
    assert apply_iterable_on_slice(gen, slice(5, 10, 1), 10) == [5, 6, 7]


#######################
# Tests for apply_slice_on_slice
#######################


def test_apply_slice_on_slice_basic():
    """Test basic functionality of apply_slice_on_slice."""
    s1 = slice(0, 10, 1)
    s2 = slice(0, 5, 1)
    result = apply_slice_on_slice(s1, s2)
    assert result.start == 0
    assert result.stop == 5
    assert result.step == 1


def test_apply_slice_on_slice_with_none():
    """Test apply_slice_on_slice when slices contain None values."""
    s1 = slice(None, None, None)
    s2 = slice(0, 5, 1)
    result = apply_slice_on_slice(s1, s2)
    assert result.start == 0
    assert result.stop == 5
    assert result.step == 1

    s1 = slice(0, 5, 1)
    s2 = slice(None, None, None)
    result = apply_slice_on_slice(s1, s2)
    assert result.start == 0
    assert result.stop == 5
    assert result.step == 1

    s1 = slice(None, 5, None)
    s2 = slice(None, None, 2)
    result = apply_slice_on_slice(s1, s2)
    assert result.start == None
    assert result.stop == 10
    assert result.step == 2


def test_apply_slice_on_slice_with_steps():
    """Test apply_slice_on_slice with different step values."""
    s1 = slice(0, 10, 2)
    s2 = slice(0, 5, 3)
    result = apply_slice_on_slice(s1, s2)
    assert result.start == 0
    assert result.stop == 5
    assert result.step == 6


def test_apply_slice_on_slice_with_offsets():
    """Test apply_slice_on_slice with offsets."""
    s1 = slice(2, 7, 1)
    s2 = slice(3, 10, 1)
    result = apply_slice_on_slice(s1, s2)
    assert result.start == 5
    assert result.stop == 10
    assert result.step == 1


def test_apply_slice_on_slice_identity():
    """Test apply_slice_on_slice with identity slices."""
    s1 = slice(None)
    s2 = slice(0, 10, 1)
    result = apply_slice_on_slice(s1, s2)
    assert result.start == 0
    assert result.stop == 10
    assert result.step == 1

    s1 = slice(0, 10, 1)
    s2 = slice(None)
    result = apply_slice_on_slice(s1, s2)
    assert result.start == 0
    assert result.stop == 10
    assert result.step == 1


#######################
# Tests for apply_slice
#######################


def test_apply_slice_slice_slice():
    """Test apply_slice with two slices."""
    s1 = slice(0, 10, 1)
    s2 = slice(0, 5, 1)
    result = apply_slice(s1, s2, 10)
    assert isinstance(result, slice)
    assert result.start == 0
    assert result.stop == 5
    assert result.step == 1


def test_apply_slice_iterable_slice():
    """Test apply_slice with an iterable and a slice."""
    s1 = [0, 1, 2]
    s2 = slice(0, 5, 1)
    result = apply_slice(s1, s2, 10)
    assert result == [0, 1, 2]

    s1 = [0, 1, 2]
    s2 = slice(5, 10, 1)
    result = apply_slice(s1, s2, 10)
    assert result == [5, 6, 7]


def test_apply_slice_slice_iterable():
    """Test apply_slice with a slice and an iterable."""
    s1 = slice(0, 2, 1)
    s2 = [5, 6, 7, 8, 9]
    result = apply_slice(s1, s2, 5)
    assert result == [5, 6]

    s1 = slice(1, 3, 1)
    s2 = [5, 6, 7, 8, 9]
    result = apply_slice(s1, s2, 5)
    assert result == [6, 7]


def test_apply_slice_iterable_iterable():
    """Test apply_slice with two iterables."""
    s1 = [0, 2, 4]
    s2 = [5, 6, 7, 8, 9]
    result = apply_slice(s1, s2, 5)
    assert result == [5, 7, 9]

    s1 = [1, 3]
    s2 = [5, 6, 7, 8, 9]
    result = apply_slice(s1, s2, 5)
    assert result == [6, 8]


def test_apply_slice_identity():
    """Test apply_slice with identity slices."""
    s1 = slice(None)
    s2 = slice(0, 10, 1)
    result = apply_slice(s1, s2, 10)
    assert result == s2

    s1 = slice(0, 10, 1)
    s2 = slice(None)
    result = apply_slice(s1, s2, 10)
    assert result == s1

    s1 = slice(None)
    s2 = [0, 1, 2, 3, 4]
    result = apply_slice(s1, s2, 5)
    assert result == s2

    s1 = [0, 1, 2, 3, 4]
    s2 = slice(None)
    result = apply_slice(s1, s2, 5)
    assert result == s1


def test_apply_slice_invalid_types():
    """Test apply_slice with invalid input types."""
    with pytest.raises(ValueError):
        apply_slice(1, 2, 10)

    with pytest.raises(TypeError):
        apply_slice("string", slice(0, 5, 1), 10)


def test_apply_slice_with_generators():
    """Test apply_slice with generators."""
    s1 = (i for i in range(3))
    s2 = slice(5, 10, 1)
    result = apply_slice(s1, s2, 10)
    assert result == [5, 6, 7]

    s1 = slice(0, 2, 1)
    s2 = (i for i in range(5, 10))
    result = apply_slice(s1, s2, 5)
    assert result == [5, 6]

    s1 = (i for i in range(0, 3, 2))
    s2 = (i for i in range(5, 10))
    result = apply_slice(s1, s2, 5)
    assert result == [5, 7]


def test_apply_slice_negative_indices():
    """Test apply_slice with negative indices."""
    s1 = slice(-5, None, None)
    s2 = slice(0, 10, 1)
    result = apply_slice(s1, s2, 10)
    assert isinstance(result, slice)

    s1 = [-1, -2, -3]
    s2 = slice(0, 10, 1)
    result = apply_slice(s1, s2, 10)
    assert result == [9, 8, 7]

    s1 = slice(0, 3, 1)
    s2 = [-1, -2, -3, -4, -5]
    result = apply_slice(s1, s2, 5)
    assert result == [-1, -2, -3]
