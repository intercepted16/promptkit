"""Type casting utilities with runtime validation."""

from typing import Any, TypeVar, cast, Type, Union, Tuple

T = TypeVar("T")


def checked_cast(
    expected_type: Union[Type[T], Tuple[Type[Any], ...]], value: Any
) -> T:
    """Cast value to expected type with runtime validation.

    Args:
        expected_type: Type or tuple of types to cast to.
        value: Value to cast.

    Returns:
        The casted value.

    Raises:
        TypeError: If value is not of expected type.
    """
    if not isinstance(value, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(value)}")
    return cast(T, value)
