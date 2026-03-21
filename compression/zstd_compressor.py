"""Plain ZSTD compression."""

from typing import Optional

import zstandard as zstd

from .base import Compressor, CompressionResult
from .registry import register_compressor


@register_compressor("zstd")
class ZstdCompressor(Compressor):
    name = "ZSTD"

    def __init__(self, level: int = 3, **kwargs):
        self.level = level

    def compress(self, filepath: str) -> Optional[CompressionResult]:
        with open(filepath, "rb") as f:
            data = f.read()

        original_size = len(data)
        cctx = zstd.ZstdCompressor(level=self.level)
        compressed_size = len(cctx.compress(data))
        return CompressionResult(original_size=original_size, compressed_size=compressed_size)
