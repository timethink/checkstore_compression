"""
Extract tensors from all checkpoints into per-parameter container files.

For each parameter and tensor type (weight, momentum, etc.), creates a
safetensors file with shape [steps, elements] containing the tensor
collected across all checkpoints.

Usage:
    python extract_containers.py --config config/flame_moe.yaml --output_dir ./containers
"""

import argparse
import os
import sys

import yaml
import torch
from safetensors.torch import save_file
from tqdm import tqdm

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ckpt_formats import get_adapter
# Import concrete adapters so they register themselves
import ckpt_formats.megatron  # noqa: F401
import ckpt_formats.huggingface  # noqa: F401


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_all(config: dict, output_dir: str):
    fmt = config["format"]
    adapter = get_adapter(fmt, config)

    checkpoints = adapter.discover_checkpoints()
    if not checkpoints:
        print("No checkpoints found. Check your config.")
        return

    print(f"Found {len(checkpoints)} checkpoints "
          f"(steps {checkpoints[0][0]}..{checkpoints[-1][0]})")

    param_names = adapter.get_parameter_names()
    tensor_types = adapter.get_tensor_types()
    print(f"Found {len(param_names)} parameters, "
          f"tensor types: {tensor_types}")

    os.makedirs(output_dir, exist_ok=True)

    for param_name in tqdm(param_names, desc="Parameters"):
        safe_name = param_name.replace("/", "_").replace(".", "_")
        param_dir = os.path.join(output_dir, safe_name)
        os.makedirs(param_dir, exist_ok=True)

        for ttype in tensor_types:
            output_file = os.path.join(param_dir, f"{ttype}.safetensors")

            # Skip if already extracted
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                continue

            # Check if this tensor type is applicable for this param
            tensor_key = adapter.get_tensor_key(param_name, ttype)
            if tensor_key is None:
                continue

            # Collect across checkpoints
            tensors_to_save = {}
            for step_num, ckpt_path in checkpoints:
                tensor = adapter.load_tensor(ckpt_path, param_name, ttype)
                if tensor is not None:
                    tensors_to_save[f"step_{step_num}"] = tensor

            if not tensors_to_save:
                continue

            try:
                save_file(tensors_to_save, output_file)
            except Exception as e:
                print(f"\nFailed to save {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract checkpoint tensors into per-parameter containers."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. config/flame_moe.yaml)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./extracted_containers",
        help="Directory to write container safetensors files"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    extract_all(config, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
