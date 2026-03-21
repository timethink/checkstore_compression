"""Compressor registry — maps method names to compressor classes."""

from typing import Dict, List, Type
from .base import Compressor

_REGISTRY: Dict[str, Type[Compressor]] = {}


def register_compressor(name: str):
    """Decorator to register a compressor class under *name*."""
    def decorator(cls: Type[Compressor]):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_compressor(name: str, **kwargs) -> Compressor:
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys()) or "(none)"
        raise ValueError(
            f"Unknown compressor '{name}'. Available: {available}"
        )
    return _REGISTRY[name](**kwargs)


def get_all_compressors(**kwargs) -> List[Compressor]:
    return [cls(**kwargs) for cls in _REGISTRY.values()]
