"""
Quick smoke test using real Flame-MoE checkpoints.

Only extracts the first 2 parameters (weight only) to keep it fast,
then runs compression on the resulting containers.

Usage:
    python test_real_data.py
"""

import json
import os
import sys
import tempfile
import shutil

import yaml
import torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ckpt_formats import get_adapter
import ckpt_formats.megatron  # noqa: F401

from compression import get_compressor
import compression.zstd_compressor  # noqa: F401
import compression.zstd_bytegrouping  # noqa: F401
import compression.zipnn_compressor  # noqa: F401


CKPT_BASE = "/mnt/sda1/yxz/Flame-moe/converted_checkpoints"
ZIPNN_SCRIPT = "/mnt/sda1/yxz/zipnn/scripts/zipnn_compress_safetensors.py"
NUM_PARAMS = 2          # only test first N parameters
TENSOR_TYPES = ["weight"]  # only weight to keep it quick


def main():
    # ---- Load config inline (same as flame_moe.yaml but minimal) ----
    config = {
        "checkpoint": {
            "base_dir": CKPT_BASE,
            "pattern": "iter_*",
            "step_regex": r"iter_(\d+)",
        },
        "format": "megatron",
        "tensor_types": {
            "weight": {
                "source": "model",
                "key_template": "{param_name}",
            },
        },
        "megatron": {
            "chained_prefix_default": "chained_0",
            "chained_prefix_expert": "chained_1",
            "chained_test_template": "chained_1.optimizer.state.exp_avg.{param_name}",
        },
    }

    adapter = get_adapter("megatron", config)

    # ---- Discover checkpoints ----
    checkpoints = adapter.discover_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints: "
          f"steps {[s for s, _ in checkpoints]}")

    # ---- Pick first N params ----
    all_params = adapter.get_parameter_names()
    test_params = all_params[:NUM_PARAMS]
    print(f"Testing {len(test_params)} params: {test_params}")

    # ---- Create temp output dir ----
    tmp_dir = tempfile.mkdtemp(prefix="flame_test_")
    print(f"Temp dir: {tmp_dir}")

    # ---- Build compressors ----
    compressors = [
        get_compressor("zstd", level=3),
        get_compressor("zstd_bytegrouping", level=3),
        get_compressor("zipnn", script_path=ZIPNN_SCRIPT),
    ]

    try:
        for param_name in test_params:
            print(f"\n{'='*60}")
            print(f"Parameter: {param_name}")
            print("=" * 60)

            for ttype in TENSOR_TYPES:
                tensor_key = adapter.get_tensor_key(param_name, ttype)
                if tensor_key is None:
                    print(f"  [{ttype}] not applicable, skip")
                    continue

                # Collect across checkpoints
                container = {}
                for step, ckpt_path in checkpoints:
                    t = adapter.load_tensor(ckpt_path, param_name, ttype)
                    if t is not None:
                        container[f"step_{step}"] = t

                if not container:
                    print(f"  [{ttype}] no data found across checkpoints")
                    continue

                # Show tensor info
                sample = list(container.values())[0]
                print(f"  [{ttype}] collected {len(container)} steps, "
                      f"shape={list(sample.shape)}, dtype={sample.dtype}")

                # Save container
                safe_name = param_name.replace("/", "_").replace(".", "_")
                container_path = os.path.join(tmp_dir, f"{safe_name}_{ttype}.safetensors")
                save_file(container, container_path)
                file_size = os.path.getsize(container_path)
                print(f"  [{ttype}] container size: {file_size / 1024:.1f} KB")

                # ---- Compress original layout ----
                print(f"  [{ttype}] Original layout:")
                for c in compressors:
                    result = c.compress(container_path)
                    if result:
                        print(f"    {c.name}: ratio={result.ratio:.4f} "
                              f"({result.compressed_size / 1024:.1f} KB)")

                # ---- Rearrange and compress ----
                tensors_loaded = load_file(container_path)
                step_keys = sorted(
                    [(int(k.split("_")[1]), k) for k in tensors_loaded if k.startswith("step_")]
                )
                stacked = torch.stack([tensors_loaded[k] for _, k in step_keys])
                num_steps = stacked.shape[0]
                flat = stacked.reshape(num_steps, -1)
                rearranged = flat.t().contiguous()

                rearranged_path = os.path.join(tmp_dir, f"{safe_name}_{ttype}_rearranged.safetensors")
                save_file({"data": rearranged}, rearranged_path)
                rearranged_size = os.path.getsize(rearranged_path)
                print(f"  [{ttype}] Rearranged layout (Elements x Steps):")
                for c in compressors:
                    result = c.compress(rearranged_path)
                    if result:
                        print(f"    {c.name}: ratio={result.ratio:.4f} "
                              f"({result.compressed_size / 1024:.1f} KB)")

        print(f"\n{'='*60}")
        print("SMOKE TEST PASSED")
        print("=" * 60)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[cleanup] Removed {tmp_dir}")


if __name__ == "__main__":
    main()
