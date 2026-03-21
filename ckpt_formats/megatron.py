"""Adapter for Megatron-style checkpoints (e.g. Flame-MoE)."""

import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open

from .base import CheckpointAdapter
from .registry import register_adapter


@register_adapter("megatron")
class MegatronAdapter(CheckpointAdapter):
    """
    Expects a directory of checkpoints like:
        base_dir/iter_00001/
        base_dir/iter_00002/
    Each containing model.safetensors.index.json and
    optionally optimizer.safetensors.index.json.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        ckpt_cfg = config["checkpoint"]
        self.base_dir = ckpt_cfg["base_dir"]
        self.pattern = ckpt_cfg.get("pattern", "iter_*")
        self.step_regex = ckpt_cfg.get("step_regex", r"iter_(\d+)")

        fmt_cfg = config.get("megatron", {})
        self.chained_default = fmt_cfg.get("chained_prefix_default", "chained_0")
        self.chained_expert = fmt_cfg.get("chained_prefix_expert", "chained_1")
        self.chained_test_tpl = fmt_cfg.get(
            "chained_test_template",
            "chained_1.optimizer.state.exp_avg.{param_name}",
        )

        self.tensor_types_cfg: Dict[str, dict] = config.get("tensor_types", {})

        # Lazily loaded index caches
        self._model_weight_map: Optional[Dict[str, str]] = None
        self._opt_weight_map: Optional[Dict[str, str]] = None
        self._param_names: Optional[List[str]] = None
        self._checkpoints: Optional[List[Tuple[int, str]]] = None

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

    def _load_index(self, filename: str) -> Dict[str, str]:
        path = os.path.join(self._first_ckpt(), filename)
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return json.load(f).get("weight_map", {})

    @property
    def model_weight_map(self) -> Dict[str, str]:
        if self._model_weight_map is None:
            self._model_weight_map = self._load_index("model.safetensors.index.json")
        return self._model_weight_map

    @property
    def opt_weight_map(self) -> Dict[str, str]:
        if self._opt_weight_map is None:
            self._opt_weight_map = self._load_index("optimizer.safetensors.index.json")
        return self._opt_weight_map

    def get_parameter_names(self) -> List[str]:
        if self._param_names is None:
            self._param_names = list(self.model_weight_map.keys())
        return self._param_names

    # ------------------------------------------------------------------
    # Tensor types
    # ------------------------------------------------------------------
    def get_tensor_types(self) -> List[str]:
        return list(self.tensor_types_cfg.keys())

    def _chained_prefix(self, param_name: str) -> str:
        test_key = self.chained_test_tpl.format(param_name=param_name)
        if test_key in self.opt_weight_map:
            return self.chained_expert
        return self.chained_default

    def get_tensor_key(self, param_name: str, tensor_type: str) -> Optional[str]:
        type_cfg = self.tensor_types_cfg.get(tensor_type)
        if type_cfg is None:
            return None
        template = type_cfg["key_template"]
        chained_prefix = self._chained_prefix(param_name)
        return template.format(param_name=param_name, chained_prefix=chained_prefix)

    def _weight_map_for_type(self, tensor_type: str) -> Dict[str, str]:
        type_cfg = self.tensor_types_cfg.get(tensor_type, {})
        source = type_cfg.get("source", "model")
        if source == "optimizer":
            return self.opt_weight_map
        return self.model_weight_map

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_tensor(
        self, ckpt_path: str, param_name: str, tensor_type: str
    ) -> Optional[torch.Tensor]:
        tensor_key = self.get_tensor_key(param_name, tensor_type)
        if tensor_key is None:
            return None

        wmap = self._weight_map_for_type(tensor_type)
        if tensor_key not in wmap:
            return None

        target_file = os.path.join(ckpt_path, wmap[tensor_key])
        if not os.path.exists(target_file):
            return None

        try:
            with safe_open(target_file, framework="pt", device="cpu") as f:
                if tensor_key in f.keys():
                    return f.get_tensor(tensor_key)
        except Exception as e:
            print(f"Error reading {target_file} key={tensor_key}: {e}")
        return None
