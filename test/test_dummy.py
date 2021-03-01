from numbers import Real

import pytest


def add_one(x: Real) -> Real:
    """
    Adds one to a real number.
    """
    if not isinstance(x, Real):
        raise ValueError(f"x type must be Real, got '{type(x).__name__}' instead.")
    return x + 1


@pytest.mark.parametrize(
    ("x, expected"), [(1, 2), (2.0, 3), (3, 4.0), (4.0, 5.0), (-1, 0.0), (-1e3, -999)]
)
def test_add_one_value(x: Real, expected: Real) -> None:
    """
    Tests the output consistency of the add_one function.
    """
    assert add_one(x) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("x, msg"),
    [
        ("a", "x type must be Real, got 'str' instead."),
        ([1, 2], "x type must be Real, got 'list' instead."),
    ],
)
def test_add_one_error(x: Real, msg: str) -> None:
    """
    Tests if add_one function correctly raises an error when its input is not a real.
    """
    with pytest.raises(ValueError, match=msg):
        add_one(x)