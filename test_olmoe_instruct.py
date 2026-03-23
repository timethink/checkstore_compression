"""
Real-world test on OLMoE-instruct checkpoints.

Picks representative parameters per type (layernorm, attention, expert, router,
embedding, etc.), extracts containers, then runs compression methods on both
original and rearranged layouts.

Model: OLMoE-instruct (HuggingFace format, bfloat16, 16 layers, 64 experts)
Steps: 44 checkpoints (step_40 to step_1760)

Usage:
    python test_olmoe_instruct.py
"""

import os
import sys
import tempfile
import shutil

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ckpt_formats import get_adapter
import ckpt_formats.huggingface  # noqa: F401

from compression import get_compressor
import compression.zstd_compressor  # noqa: F401
import compression.zstd_bytegrouping  # noqa: F401

# Representative parameters — one per type
TEST_PARAMS = [
    # Small: layernorm / norm
    "model.norm.weight",                                    # final RMSNorm
    "model.layers.0.input_layernorm.weight",                # input layernorm
    "model.layers.0.post_attention_layernorm.weight",       # post-attn layernorm
    # Medium: attention projections (2048x2048 = 8MB each)
    "model.layers.0.self_attn.q_proj.weight",               # attention Q
    "model.layers.0.self_attn.k_proj.weight",               # attention K
    "model.layers.0.self_attn.o_proj.weight",               # attention O
    # Medium: MoE expert (2048x1024 = 4MB each)
    "model.layers.0.mlp.experts.0.down_proj.weight",        # expert 0 down
    "model.layers.0.mlp.experts.0.gate_proj.weight",        # expert 0 gate
    "model.layers.0.mlp.experts.0.up_proj.weight",          # expert 0 up
    # Small: MoE router
    "model.layers.0.mlp.gate.weight",                       # router/gate
    # Large: embedding & output head (~196.5MB each)
    "model.embed_tokens.weight",                            # embedding
    "lm_head.weight",                                       # output head
]


def fmt_size(nbytes):
    if nbytes >= 1024 * 1024:
        return f"{nbytes / 1024 / 1024:.1f} MB"
    return f"{nbytes / 1024:.1f} KB"


def _int_dtype_for(tensor):
    """Return an integer dtype with the same element size for bitwise ops."""
    sz = tensor.element_size()
    return {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(sz, torch.uint8)


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


def xor_delta_original(container_path, output_path):
    """XOR each step with the previous step (original layout).

    step_0 is kept as-is; step_N = step_N XOR step_{N-1}.
    """
    tensors = load_file(container_path)
    step_keys = sorted(
        [(int(k.split("_")[1]), k) for k in tensors if k.startswith("step_")]
    )
    if len(step_keys) < 2:
        return False

    int_dtype = _int_dtype_for(tensors[step_keys[0][1]])
    result = {}
    prev_int = None
    for _, key in step_keys:
        t_int = tensors[key].contiguous().view(int_dtype)
        if prev_int is None:
            result[key] = t_int
        else:
            result[key] = torch.bitwise_xor(t_int, prev_int)
        prev_int = t_int

    save_file(result, output_path)
    return True


def xor_delta_rearranged(rearranged_path, output_path):
    """XOR along the Steps axis on rearranged [Elements, Steps] data.

    Column 0 is kept as-is; column_t = column_t XOR column_{t-1}.
    """
    tensors = load_file(rearranged_path)
    data = tensors["data"]  # [Elements, Steps]
    int_dtype = _int_dtype_for(data)
    d = data.contiguous().view(int_dtype)

    result = torch.empty_like(d)
    result[:, 0] = d[:, 0]
    result[:, 1:] = torch.bitwise_xor(d[:, 1:], d[:, :-1])

    save_file({"data": result}, output_path)
    return True


def main():
    config = {
        "checkpoint": {
            "base_dir": "/mnt/sda1/yxz/OLMoE-instruct_Downloads",
            "pattern": "step_*",
            "step_regex": r"step_(\d+)",
        },
        "format": "huggingface",
        "tensor_types": {
            "weight": {"source": "model", "key_template": "{param_name}"},
        },
    }

    adapter = get_adapter("huggingface", config)
    checkpoints = adapter.discover_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints: "
          f"steps {checkpoints[0][0]}..{checkpoints[-1][0]}")

    compressors = [
        get_compressor("zstd", level=3),
        get_compressor("zstd_bytegrouping", level=3),
    ]
    zstd = compressors[0]  # reuse for XOR delta tests

    tmp_dir = tempfile.mkdtemp(prefix="olmoe_test_")
    print(f"Temp dir: {tmp_dir}\n")

    results = []

    try:
        for param_name in TEST_PARAMS:
            print(f"{'='*70}")
            print(f"Parameter: {param_name}")
            print("=" * 70)

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

            # --- Compress original layout ---
            print(f"  Original layout:")
            for c in compressors:
                r = c.compress(container_path)
                if r:
                    print(f"    {c.name:25s}  ratio={r.ratio:.4f}  "
                          f"({fmt_size(r.compressed_size)})")
                    row[f"Orig_{c.name}"] = f"{r.ratio:.4f}"
                else:
                    row[f"Orig_{c.name}"] = "N/A"

            # --- XOR delta on original layout ---
            xor_orig_path = os.path.join(
                tmp_dir, f"{safe_name}_xor_orig.safetensors")
            if xor_delta_original(container_path, xor_orig_path):
                print(f"  XOR delta (original layout):")
                r = zstd.compress(xor_orig_path)
                if r:
                    print(f"    {'ZSTD':25s}  ratio={r.ratio:.4f}  "
                          f"({fmt_size(r.compressed_size)})")
                    row["XorOrig_ZSTD"] = f"{r.ratio:.4f}"
                else:
                    row["XorOrig_ZSTD"] = "N/A"
                os.remove(xor_orig_path)

            # --- Rearrange + compress ---
            rearranged_path = os.path.join(
                tmp_dir, f"{safe_name}_rearranged.safetensors")
            ok = rearrange(container_path, rearranged_path)
            if ok:
                print(f"  Rearranged layout (Elements x Steps):")
                for c in compressors:
                    r = c.compress(rearranged_path)
                    if r:
                        print(f"    {c.name:25s}  ratio={r.ratio:.4f}  "
                              f"({fmt_size(r.compressed_size)})")
                        row[f"Rearr_{c.name}"] = f"{r.ratio:.4f}"
                    else:
                        row[f"Rearr_{c.name}"] = "N/A"

                # --- XOR delta on rearranged layout ---
                xor_rearr_path = os.path.join(
                    tmp_dir, f"{safe_name}_xor_rearr.safetensors")
                if xor_delta_rearranged(rearranged_path, xor_rearr_path):
                    print(f"  XOR delta (rearranged layout):")
                    r = zstd.compress(xor_rearr_path)
                    if r:
                        print(f"    {'ZSTD':25s}  ratio={r.ratio:.4f}  "
                              f"({fmt_size(r.compressed_size)})")
                        row["XorRearr_ZSTD"] = f"{r.ratio:.4f}"
                    else:
                        row["XorRearr_ZSTD"] = "N/A"
                    os.remove(xor_rearr_path)

                os.remove(rearranged_path)

            os.remove(container_path)
            results.append(row)
            print()

        # --- Summary table ---
        print("\n" + "=" * 100)
        print("SUMMARY — OLMoE-instruct")
        print("=" * 100)
        header = (f"{'Parameter':<45s} {'Size':>8s} | "
                  f"{'ZSTD':>6s} {'BG':>6s} | {'ZSTD':>6s} {'BG':>6s} | "
                  f"{'XOR+Z':>6s} {'XOR+Z':>6s}")
        print(f"{'':45s} {'':>8s} | "
              f"{'-- Original --':^13s} | {'-- Rearranged -':^13s} | "
              f"{'- XOR delta -':^13s}")
        print(f"{'':45s} {'':>8s} | "
              f"{'':>6s} {'':>6s} | {'':>6s} {'':>6s} | "
              f"{'Orig':>6s} {'Rearr':>6s}")
        print(header)
        print("-" * len(header))
        for row in results:
            short = row["param"]
            if len(short) > 44:
                short = "..." + short[-41:]
            print(f"{short:<45s} {row['size']:>8s} | "
                  f"{row.get('Orig_ZSTD',''):>6s} "
                  f"{row.get('Orig_ZSTD+ByteGrouping',''):>6s} | "
                  f"{row.get('Rearr_ZSTD',''):>6s} "
                  f"{row.get('Rearr_ZSTD+ByteGrouping',''):>6s} | "
                  f"{row.get('XorOrig_ZSTD',''):>6s} "
                  f"{row.get('XorRearr_ZSTD',''):>6s}")

        print(f"\nDONE — tested {len(results)} parameters "
              f"x {len(compressors)} methods x 2 layouts + XOR delta")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[cleanup] Removed {tmp_dir}")


if __name__ == "__main__":
    main()
