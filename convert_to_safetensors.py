"""
Convert checkpoints to safetensors format using HuggingFace Transformers.

Usage:
    python convert_to_safetensors.py --src_dir amber_ckpts --dst_dir standardized_ckpts \
        --pattern "ckpt_*" --step_regex "ckpt_(\\d+)" \
        --max_shard_size 5GB --num_latest 10
"""

import argparse
import gc
import glob
import os
import re
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def discover_checkpoints(
    src_dir: str, pattern: str, step_regex: str
) -> List[Tuple[int, str]]:
    regex = re.compile(step_regex)
    subdirs = sorted(glob.glob(os.path.join(src_dir, pattern)))
    results = []
    for d in subdirs:
        if os.path.isdir(d):
            m = regex.search(os.path.basename(d))
            if m:
                results.append((int(m.group(1)), d))
    results.sort(key=lambda x: x[0])
    return results


def convert_checkpoint(
    src_path: str,
    dst_path: str,
    max_shard_size: str,
    dtype: torch.dtype,
):
    # Check if already converted
    if os.path.exists(dst_path) and (
        os.path.exists(os.path.join(dst_path, "model.safetensors"))
        or os.path.exists(os.path.join(dst_path, "model.safetensors.index.json"))
    ):
        print(f"  Already converted, skipping: {dst_path}")
        return

    print(f"  Loading model from {src_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(src_path)
    model = AutoModelForCausalLM.from_pretrained(
        src_path, torch_dtype=dtype, low_cpu_mem_usage=True
    )

    os.makedirs(dst_path, exist_ok=True)
    print(f"  Saving to {dst_path} (max_shard_size={max_shard_size}) ...")
    model.save_pretrained(
        dst_path, safe_serialization=True, max_shard_size=max_shard_size
    )
    tokenizer.save_pretrained(dst_path)

    del model, tokenizer
    gc.collect()
    print(f"  Done: {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert checkpoints to safetensors format."
    )
    parser.add_argument("--src_dir", type=str, required=True,
                        help="Directory containing original checkpoints")
    parser.add_argument("--dst_dir", type=str, required=True,
                        help="Directory to write converted checkpoints")
    parser.add_argument("--pattern", type=str, default="ckpt_*",
                        help="Glob pattern for checkpoint directories")
    parser.add_argument("--step_regex", type=str, default=r"ckpt_(\d+)",
                        help="Regex to extract step number from directory name")
    parser.add_argument("--max_shard_size", type=str, default="5GB",
                        help="Max shard size for safetensors output")
    parser.add_argument("--num_latest", type=int, default=0,
                        help="Only convert the N most recent checkpoints (0 = all)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Torch dtype for loading")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    all_ckpts = discover_checkpoints(args.src_dir, args.pattern, args.step_regex)
    if not all_ckpts:
        print("No checkpoints found.")
        return

    if args.num_latest > 0:
        all_ckpts = all_ckpts[-args.num_latest:]

    print(f"Checkpoints to convert: {[name for _, name in all_ckpts]}")

    os.makedirs(args.dst_dir, exist_ok=True)
    for _step_num, src_path in all_ckpts:
        step_name = os.path.basename(src_path)
        dst_path = os.path.join(args.dst_dir, step_name)
        print(f"\n[{step_name}]")
        try:
            convert_checkpoint(src_path, dst_path, args.max_shard_size,
                               dtype_map[args.dtype])
        except Exception as e:
            print(f"  Error converting {step_name}: {e}")


if __name__ == "__main__":
    main()
