"""Abstract base class for compression methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class CompressionResult:
    original_size: int
    compressed_size: int

    @property
    def ratio(self) -> float:
        if self.original_size == 0:
            return 0.0
        return self.compressed_size / self.original_size


class Compressor(ABC):
    """Each compression method implements this interface."""

    name: str = "base"

    @abstractmethod
    def compress(self, filepath: str) -> Optional[CompressionResult]:
        """
        Compress the given file and return the result.
        Return None if compression is not applicable or fails.
        """
        ...
