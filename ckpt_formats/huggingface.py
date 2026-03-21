"""Adapter for HuggingFace-style checkpoints (e.g. Amber, Llama, etc.)."""

import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open

from .base import CheckpointAdapter
from .registry import register_adapter


@register_adapter("huggingface")
class HuggingFaceAdapter(CheckpointAdapter):
    """
    Expects a directory of checkpoints like:
        base_dir/ckpt_100/
        base_dir/checkpoint-200/
    Each containing either a single model.safetensors or
    model.safetensors.index.json + shards.
    Only model weights are supported (no optimizer states).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        ckpt_cfg = config["checkpoint"]
        self.base_dir = ckpt_cfg["base_dir"]
        self.pattern = ckpt_cfg.get("pattern", "ckpt_*")
        self.step_regex = ckpt_cfg.get("step_regex", r"ckpt_(\d+)")

        self.tensor_types_cfg: Dict[str, dict] = config.get("tensor_types", {})

        self._checkpoints: Optional[List[Tuple[int, str]]] = None
        self._weight_map: Optional[Dict[str, str]] = None
        self._param_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_checkpoints(self) -> List[Tuple[int, str]]:
        if self._checkpoints is not None:
            return self._checkpoints

        regex = re.compile(self.step_regex)
        subdirs = sorted(glob.glob(os.path.join(self.base_dir, self.pattern)))
        results: List[Tuple[int, str]] = []
        for d in subdirs:
            m = regex.search(os.path.basename(d))
            if m:
                step = int(m.group(1))
                results.append((step, d))
        results.sort(key=lambda x: x[0])
        self._checkpoints = results
        return results

    def _first_ckpt(self) -> str:
        ckpts = self.discover_checkpoints()
        if not ckpts:
            raise RuntimeError(f"No checkpoints found in {self.base_dir}")
        return ckpts[0][1]

    def _load_weight_map(self) -> Dict[str, str]:
        first = self._first_ckpt()
        index_path = os.path.join(first, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                return json.load(f).get("weight_map", {})
        # Single-file model
        single = os.path.join(first, "model.safetensors")
        if os.path.exists(single):
            with safe_open(single, framework="pt", device="cpu") as f:
                return {k: "model.safetensors" for k in f.keys()}
        return {}

    @property
    def weight_map(self) -> Dict[str, str]:
        if self._weight_map is None:
            self._weight_map = self._load_weight_map()
        return self._weight_map

    def get_parameter_names(self) -> List[str]:
        if self._param_names is None:
            self._param_names = list(self.weight_map.keys())
        return self._param_names

    # ------------------------------------------------------------------
    # Tensor types
    # ------------------------------------------------------------------
    def get_tensor_types(self) -> List[str]:
        return list(self.tensor_types_cfg.keys())

    def get_tensor_key(self, param_name: str, tensor_type: str) -> Optional[str]:
        type_cfg = self.tensor_types_cfg.get(tensor_type)
        if type_cfg is None:
            return None
        template = type_cfg["key_template"]
        return template.format(param_name=param_name)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_tensor(
        self, ckpt_path: str, param_name: str, tensor_type: str
    ) -> Optional[torch.Tensor]:
        tensor_key = self.get_tensor_key(param_name, tensor_type)
        if tensor_key is None:
            return None

        if tensor_key not in self.weight_map:
            return None

        target_file = os.path.join(ckpt_path, self.weight_map[tensor_key])
        if not os.path.exists(target_file):
            return None

        try:
            with safe_open(target_file, framework="pt", device="cpu") as f:
                if tensor_key in f.keys():
                    return f.get_tensor(tensor_key)
        except Exception as e:
            print(f"Error reading {target_file} key={tensor_key}: {e}")
        return None
