"""ZSTD compression with byte-grouping pre-processing."""

from typing import Optional

import numpy as np
import torch
import zstandard as zstd
from safetensors.torch import load_file

from .base import Compressor, CompressionResult
from .registry import register_compressor


def _byte_group_tensor(tensor: torch.Tensor) -> bytes:
    """Re-arrange tensor bytes so that same-significance bytes are adjacent."""
    elem_size = tensor.element_size()
    arr_uint8 = tensor.flatten().view(torch.uint8).cpu().numpy()
    if elem_size > 1:
        return np.ascontiguousarray(arr_uint8.reshape(-1, elem_size).T).tobytes()
    return arr_uint8.tobytes()


@register_compressor("zstd_bytegrouping")
class ZstdByteGroupingCompressor(Compressor):
    name = "ZSTD+ByteGrouping"

    def __init__(self, level: int = 3, **kwargs):
        self.level = level

    def compress(self, filepath: str) -> Optional[CompressionResult]:
        # Read safetensors header verbatim
        with open(filepath, "rb") as f:
            header_len_bytes = f.read(8)
            header_len = np.frombuffer(header_len_bytes, dtype=np.uint64)[0]
            header_json = f.read(header_len)

        tensors = load_file(filepath)
        grouped_data = bytearray(header_len_bytes + header_json)

        for _name, tensor in tensors.items():
            try:
                grouped_data.extend(_byte_group_tensor(tensor))
            except Exception:
                grouped_data.extend(tensor.cpu().numpy().tobytes())

        data = bytes(grouped_data)
        original_size = len(data)

        cctx = zstd.ZstdCompressor(level=self.level)
        compressed_size = len(cctx.compress(data))
        return CompressionResult(original_size=original_size, compressed_size=compressed_size)
