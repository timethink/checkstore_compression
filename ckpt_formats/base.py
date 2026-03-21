"""Abstract base class for checkpoint format adapters."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch


class CheckpointAdapter(ABC):
    """
    Each checkpoint format (Megatron, HuggingFace, DeepSpeed, etc.)
    implements this interface so that the extraction pipeline is
    format-agnostic.
    """

    def __init__(self, config: dict):
        self.config = config

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    @abstractmethod
    def discover_checkpoints(self) -> List[Tuple[int, str]]:
        """Return [(step_number, checkpoint_path), ...] sorted by step."""
        ...

    @abstractmethod
    def get_parameter_names(self) -> List[str]:
        """Return all parameter names found in the first checkpoint."""
        ...

    # ------------------------------------------------------------------
    # Tensor type support
    # ------------------------------------------------------------------
    @abstractmethod
    def get_tensor_types(self) -> List[str]:
        """Return supported tensor type names, e.g. ['weight', 'momentum', ...]."""
        ...

    @abstractmethod
    def get_tensor_key(self, param_name: str, tensor_type: str) -> Optional[str]:
        """
        Given a parameter name and tensor type, return the actual key
        used inside the checkpoint file, or None if not applicable.
        """
        ...

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @abstractmethod
    def load_tensor(
        self, ckpt_path: str, param_name: str, tensor_type: str
    ) -> Optional[torch.Tensor]:
        """Load a single tensor from a checkpoint. Return None on failure."""
        ...
