"""Adapter registry — maps format names to adapter classes."""

from typing import Dict, Type
from .base import CheckpointAdapter

_REGISTRY: Dict[str, Type[CheckpointAdapter]] = {}


def register_adapter(name: str):
    """Decorator to register an adapter class under *name*."""
    def decorator(cls: Type[CheckpointAdapter]):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_adapter(name: str, config: dict) -> CheckpointAdapter:
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys()) or "(none)"
        raise ValueError(
            f"Unknown checkpoint format '{name}'. Available: {available}"
        )
    return _REGISTRY[name](config)
