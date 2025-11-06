"""Type safety utilities for handling unknown/external data structures."""

from __future__ import annotations

from typing import Any, Dict


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict-like object.

    Args:
        obj: The object to extract value from (dict-like or object with attributes)
        key: The key/attribute name to retrieve
        default: Default value to return if key is not found

    Returns:
        The value associated with the key, or default if not found
    """
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)  # type: ignore[misc]
        if hasattr(obj, key):
            return getattr(obj, key, default)  # type: ignore[misc]
    except (AttributeError, TypeError):
        pass
    return default


def ensure_dict(obj: Any) -> Dict[str, Any]:
    """Ensure an object is converted to a dict with proper typing.

    Args:
        obj: Object to convert to dict (if possible)

    Returns:
        A properly typed dictionary, or empty dict if conversion fails
    """
    if isinstance(obj, dict):
        return dict(obj)  # type: ignore[misc]
    return {}


def as_dict(obj: Any) -> Dict[str, Any]:
    """Convert an object to a dictionary using various methods.

    Tries multiple approaches to convert objects to dictionaries:
    - Direct dict conversion
    - Pydantic model_dump()
    - Legacy dict() method
    - Object __dict__ attribute

    Args:
        obj: Object to convert to dictionary

    Returns:
        Dictionary representation of the object, or empty dict if conversion fails
    """
    if isinstance(obj, dict):
        return dict(obj)  # type: ignore[misc]

    # Try pydantic model_dump
    try:
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump", None)):
            result = obj.model_dump()  # type: ignore[misc]
            if isinstance(result, dict):
                return dict(result)  # type: ignore[misc]
    except Exception:
        pass

    # Try legacy dict method
    try:
        if hasattr(obj, "dict") and callable(getattr(obj, "dict", None)):
            result = obj.dict()  # type: ignore[misc]
            if isinstance(result, dict):
                return dict(result)  # type: ignore[misc]
    except Exception:
        pass

    # Try __dict__
    try:
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)  # type: ignore[misc]
    except Exception:
        pass

    # Fallback to empty dict
    return {}
