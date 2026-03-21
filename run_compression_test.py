"""
Run compression tests on extracted container safetensors files.

For each container file, optionally rearranges to (Elements, Steps) layout,
then tests all configured compression methods on both layouts.

Usage:
    python run_compression_test.py --config config/flame_moe.yaml \
        --input_dir ./extracted_containers --output_csv results.csv
"""

import argparse
import json
import os
import sys
import traceback

import yaml
import torch
import pandas as pd
from safetensors.torch import load_file, save_file
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compression import get_compressor
# Import concrete compressors so they register themselves
import compression.zstd_compressor  # noqa: F401
import compression.zstd_bytegrouping  # noqa: F401
import compression.zipnn_compressor  # noqa: F401


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =========================================================================
# Rearrange: [Steps, *Shape] -> single tensor of shape [Elements, Steps]
# =========================================================================
def rearrange_container(input_path: str, output_path: str) -> bool:
    """Transpose a container from (Steps, Elements) to (Elements, Steps)."""
    tensors = load_file(input_path)

    # Sort step keys
    step_keys = []
    for k in tensors:
        if k.startswith("step_"):
            try:
                step_keys.append((int(k.split("_")[1]), k))
            except ValueError:
                pass
    step_keys.sort()

    if not step_keys:
        return False

    sorted_tensors = [tensors[k] for _, k in step_keys]
    step_ids = [s for s, _ in step_keys]

    stacked = torch.stack(sorted_tensors)  # [Steps, *Shape]
    num_steps = stacked.shape[0]
    original_shape = list(stacked.shape[1:])
    flat = stacked.reshape(num_steps, -1)  # [Steps, Elements]
    rearranged = flat.t().contiguous()     # [Elements, Steps]

    metadata = {
        "num_steps": str(num_steps),
        "step_ids": json.dumps(step_ids),
        "original_shape": json.dumps(original_shape),
        "layout": "elements_x_steps",
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_file({"data": rearranged}, output_path, metadata=metadata)
    return True


# =========================================================================
# Build compressor instances from config
# =========================================================================
def build_compressors(config: dict):
    comp_cfg = config.get("compression", {})
    methods_cfg = comp_cfg.get("methods", [])

    compressors = []
    for m in methods_cfg:
        name = m.pop("name")
        try:
            c = get_compressor(name, **m)
            compressors.append(c)
        except ValueError as e:
            print(f"Warning: {e}")
        finally:
            m["name"] = name  # restore for potential re-use
    return compressors


# =========================================================================
# Test a single file with all compressors
# =========================================================================
def test_file(filepath: str, compressors, tag: str) -> dict:
    metrics = {}
    for c in compressors:
        try:
            result = c.compress(filepath)
            if result is not None:
                key = f"{tag}_{c.name}"
                metrics[key] = result.ratio
                print(f"  [{c.name}] {tag}: "
                      f"{result.compressed_size / 1024**2:.2f} MB, "
                      f"ratio={result.ratio:.4f}")
            else:
                metrics[f"{tag}_{c.name}"] = None
        except Exception as e:
            print(f"  [{c.name}] {tag} failed: {e}")
            metrics[f"{tag}_{c.name}"] = None
    return metrics


# =========================================================================
# Process a single container file
# =========================================================================
def process_single_file(filepath: str, compressors, skip_rearrange: bool = False):
    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")

    row = {"File": filepath}

    # Original layout
    print("  >> Original layout")
    row.update(test_file(filepath, compressors, "Orig"))

    # Rearranged layout
    if not skip_rearrange:
        parent = os.path.dirname(filepath)
        base = os.path.basename(filepath)
        rearranged_path = os.path.join(parent, f"_rearranged_{base}")
        try:
            ok = rearrange_container(filepath, rearranged_path)
            if ok:
                print("  >> Rearranged layout (Elements x Steps)")
                row.update(test_file(rearranged_path, compressors, "Rearranged"))
            else:
                print("  >> Rearrange skipped (no step keys found)")
        except Exception as e:
            print(f"  >> Rearrange failed: {e}")
            traceback.print_exc()
        finally:
            if os.path.exists(rearranged_path):
                os.remove(rearranged_path)

    return row


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run compression tests on container safetensors files."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing extracted container safetensors"
    )
    parser.add_argument(
        "--output_csv", type=str, default="compression_results.csv",
        help="Path for the output CSV"
    )
    parser.add_argument(
        "--skip_rearrange", action="store_true",
        help="Skip the rearrangement step"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    compressors = build_compressors(config)
    if not compressors:
        print("No compression methods configured. Check your config.")
        return

    print(f"Compression methods: {[c.name for c in compressors]}")

    # Discover all safetensors files
    import glob
    files = sorted(glob.glob(
        os.path.join(args.input_dir, "**", "*.safetensors"), recursive=True
    ))
    # Exclude rearranged temp files
    files = [f for f in files if not os.path.basename(f).startswith("_rearranged_")]

    if not files:
        print(f"No safetensors files found in {args.input_dir}")
        return

    # Resume support: load existing CSV
    existing_files = set()
    existing_df = None
    if os.path.exists(args.output_csv):
        try:
            existing_df = pd.read_csv(args.output_csv)
            existing_files = set(existing_df["File"].tolist())
            print(f"Loaded {len(existing_files)} existing results, will skip those.")
        except Exception:
            pass

    new_results = []
    for fp in tqdm(files, desc="Files"):
        if fp in existing_files:
            continue
        row = process_single_file(fp, compressors, args.skip_rearrange)
        if row:
            new_results.append(row)

    if new_results:
        new_df = pd.DataFrame(new_results)
        if existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df
        df.to_csv(args.output_csv, index=False)
        print(f"\n{len(new_results)} new results. "
              f"Total {len(df)} saved to {args.output_csv}")
    else:
        print("No new files to process.")


if __name__ == "__main__":
    main()
