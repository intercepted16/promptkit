"""Dataclass field helper for default factories."""

from dataclasses import field as _field
from typing import Any


def field(*, default_factory: Any, **kwargs: Any) -> Any:
    """Wrapper around dataclasses.field to handle type checking."""
    # bury the arg-type mismatch here once
    return _field(default_factory=default_factory, **kwargs)  # type: ignore[arg-type]
