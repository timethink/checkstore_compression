"""
Real-world test on Flame-MoE checkpoints.

Picks one representative parameter per type (small layernorm, large linear,
MoE expert, router, embedding, etc.), extracts containers, then runs all
three compression methods on both original and rearranged layouts.

Usage:
    python test_flame_moe_full.py
"""

import json
import os
import sys
import tempfile
import shutil

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ckpt_formats import get_adapter
import ckpt_formats.megatron  # noqa: F401

from compression import get_compressor
import compression.zstd_compressor  # noqa: F401
import compression.zstd_bytegrouping  # noqa: F401
import compression.zipnn_compressor  # noqa: F401


CKPT_BASE = "/mnt/sda1/yxz/Flame-moe/converted_checkpoints"
ZIPNN_SCRIPT = "/mnt/sda1/yxz/zipnn/scripts/zipnn_compress_safetensors.py"

# Representative parameters — one per type
TEST_PARAMS = [
    "decoder.final_layernorm.weight",                              # small layernorm
    "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",# small layernorm (qkv)
    "decoder.layers.0.self_attention.linear_qkv.weight",           # large: attention qkv
    "decoder.layers.0.self_attention.linear_proj.weight",           # large: attention proj
    "decoder.layers.0.mlp.linear_fc1.weight",                      # large: dense MLP fc1
    "decoder.layers.0.mlp.linear_fc2.weight",                      # large: dense MLP fc2
    "decoder.layers.1.mlp.experts.experts.linear_fc1.weight",      # large: MoE expert fc1
    "decoder.layers.1.mlp.experts.experts.linear_fc2.weight",      # large: MoE expert fc2
    "decoder.layers.1.mlp.router.weight",                          # small: MoE router
    "decoder.layers.1.pre_mlp_layernorm.weight",                   # small: pre-MLP layernorm
    "embedding.word_embeddings.weight",                             # embedding
    "output_layer.weight",                                          # output head
]


def fmt_size(nbytes):
    if nbytes >= 1024 * 1024:
        return f"{nbytes / 1024 / 1024:.1f} MB"
    return f"{nbytes / 1024:.1f} KB"


def rearrange(container_path, output_path):
    tensors = load_file(container_path)
    step_keys = sorted(
        [(int(k.split("_")[1]), k) for k in tensors if k.startswith("step_")]
    )
    if not step_keys:
        return False
    stacked = torch.stack([tensors[k] for _, k in step_keys])
    num_steps = stacked.shape[0]
    flat = stacked.reshape(num_steps, -1)
    rearranged = flat.t().contiguous()
    save_file({"data": rearranged}, output_path)
    return True


def main():
    config = {
        "checkpoint": {
            "base_dir": CKPT_BASE,
            "pattern": "iter_*",
            "step_regex": r"iter_(\d+)",
        },
        "format": "megatron",
        "tensor_types": {
            "weight": {"source": "model", "key_template": "{param_name}"},
        },
        "megatron": {
            "chained_prefix_default": "chained_0",
            "chained_prefix_expert": "chained_1",
            "chained_test_template": "chained_1.optimizer.state.exp_avg.{param_name}",
        },
    }

    adapter = get_adapter("megatron", config)
    checkpoints = adapter.discover_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints: steps {[s for s, _ in checkpoints]}")

    compressors = [
        get_compressor("zstd", level=3),
        get_compressor("zstd_bytegrouping", level=3),
        get_compressor("zipnn", script_path=ZIPNN_SCRIPT),
    ]

    tmp_dir = tempfile.mkdtemp(prefix="flame_full_test_")
    print(f"Temp dir: {tmp_dir}\n")

    # Collect results for summary table
    results = []

    try:
        for param_name in TEST_PARAMS:
            print(f"{'='*70}")
            print(f"Parameter: {param_name}")
            print("=" * 70)

            # --- Extract ---
            tensor_key = adapter.get_tensor_key(param_name, "weight")
            if tensor_key is None:
                print("  SKIP: tensor key not found\n")
                continue

            container = {}
            for step, ckpt_path in checkpoints:
                t = adapter.load_tensor(ckpt_path, param_name, "weight")
                if t is not None:
                    container[f"step_{step}"] = t

            if not container:
                print("  SKIP: no data across checkpoints\n")
                continue

            sample = list(container.values())[0]
            print(f"  Shape: {list(sample.shape)}, Dtype: {sample.dtype}, "
                  f"Steps: {len(container)}")

            safe_name = param_name.replace("/", "_").replace(".", "_")
            container_path = os.path.join(tmp_dir, f"{safe_name}.safetensors")
            save_file(container, container_path)
            orig_size = os.path.getsize(container_path)
            print(f"  Container size: {fmt_size(orig_size)}")

            row = {"param": param_name, "shape": str(list(sample.shape)),
                   "dtype": str(sample.dtype), "size": fmt_size(orig_size)}

            # --- Compress original ---
            print(f"  Original layout:")
            for c in compressors:
                r = c.compress(container_path)
                if r:
                    print(f"    {c.name:20s}  ratio={r.ratio:.4f}  ({fmt_size(r.compressed_size)})")
                    row[f"Orig_{c.name}"] = f"{r.ratio:.4f}"
                else:
                    row[f"Orig_{c.name}"] = "N/A"

            # --- Rearrange + compress ---
            rearranged_path = os.path.join(tmp_dir, f"{safe_name}_rearranged.safetensors")
            ok = rearrange(container_path, rearranged_path)
            if ok:
                print(f"  Rearranged layout (Elements x Steps):")
                for c in compressors:
                    r = c.compress(rearranged_path)
                    if r:
                        print(f"    {c.name:20s}  ratio={r.ratio:.4f}  ({fmt_size(r.compressed_size)})")
                        row[f"Rearr_{c.name}"] = f"{r.ratio:.4f}"
                    else:
                        row[f"Rearr_{c.name}"] = "N/A"
                os.remove(rearranged_path)

            # Clean up container to save disk
            os.remove(container_path)
            results.append(row)
            print()

        # --- Summary table ---
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        header = f"{'Parameter':<55s} {'Size':>8s} | {'ZSTD':>6s} {'BG':>6s} {'ZNN':>6s} | {'ZSTD':>6s} {'BG':>6s} {'ZNN':>6s}"
        print(f"{'':55s} {'':>8s} | {'--- Original ---':^20s} | {'--- Rearranged ---':^20s}")
        print(header)
        print("-" * len(header))
        for row in results:
            short_name = row["param"].replace("decoder.", "").replace("layers.", "L")
            if len(short_name) > 54:
                short_name = "..." + short_name[-51:]
            print(f"{short_name:<55s} {row['size']:>8s} | "
                  f"{row.get('Orig_ZSTD',''):>6s} "
                  f"{row.get('Orig_ZSTD+ByteGrouping',''):>6s} "
                  f"{row.get('Orig_ZipNN',''):>6s} | "
                  f"{row.get('Rearr_ZSTD',''):>6s} "
                  f"{row.get('Rearr_ZSTD+ByteGrouping',''):>6s} "
                  f"{row.get('Rearr_ZipNN',''):>6s}")

        print(f"\nDONE — tested {len(results)} parameters x 3 methods x 2 layouts")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[cleanup] Removed {tmp_dir}")


if __name__ == "__main__":
    main()
