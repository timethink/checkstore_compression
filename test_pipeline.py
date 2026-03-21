"""
End-to-end test for the checkpoint compression toolkit.

Creates fake checkpoints in a temp directory, then runs:
  1. extract_containers.py  — extract containers from fake checkpoints
  2. run_compression_test.py — compress containers and produce CSV

Usage:
    python test_pipeline.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import save_file


# =========================================================================
# Helper: create fake HuggingFace-style checkpoints
# =========================================================================
def create_fake_hf_checkpoints(base_dir: str, num_steps: int = 3):
    """
    Create a minimal set of fake HF-style sharded checkpoints.
    Each checkpoint has 2 small parameters spread across 1 shard file.
    """
    param_names = [
        "model.layers.0.weight",
        "model.layers.1.weight",
    ]
    shard_file = "model-00001-of-00001.safetensors"

    # Build weight_map
    weight_map = {p: shard_file for p in param_names}

    for step in range(1, num_steps + 1):
        ckpt_dir = os.path.join(base_dir, f"ckpt_{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Write index
        index = {
            "metadata": {"total_size": 0},
            "weight_map": weight_map,
        }
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)

        # Write shard with small tensors (values drift slightly per step)
        tensors = {}
        for p in param_names:
            tensors[p] = torch.randn(4, 8) + step * 0.01
        save_file(tensors, os.path.join(ckpt_dir, shard_file))

    print(f"[setup] Created {num_steps} fake HF checkpoints in {base_dir}")
    return param_names


# =========================================================================
# Helper: write a minimal YAML config
# =========================================================================
def write_test_config(config_path: str, ckpt_dir: str):
    """Write a YAML config for the fake HF checkpoints (no zipnn)."""
    yaml_content = f"""\
checkpoint:
  base_dir: "{ckpt_dir}"
  pattern: "ckpt_*"
  step_regex: "ckpt_(\\\\d+)"

format: huggingface

tensor_types:
  weight:
    source: model
    key_template: "{{param_name}}"

compression:
  methods:
    - name: zstd
      level: 3
    - name: zstd_bytegrouping
      level: 3
"""
    with open(config_path, "w") as f:
        f.write(yaml_content)
    print(f"[setup] Wrote test config to {config_path}")


# =========================================================================
# Run a script as a subprocess and stream output
# =========================================================================
def run_script(script: str, args: list[str], label: str) -> bool:
    cmd = [sys.executable, script] + args
    print(f"\n{'='*60}")
    print(f"[{label}] Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"[{label}] FAILED (exit code {result.returncode})")
        return False

    print(f"[{label}] OK")
    return True


# =========================================================================
# Main test
# =========================================================================
def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    tmp_root = tempfile.mkdtemp(prefix="ckpt_test_")
    print(f"[setup] Temp directory: {tmp_root}")

    ckpt_dir = os.path.join(tmp_root, "checkpoints")
    container_dir = os.path.join(tmp_root, "containers")
    config_path = os.path.join(tmp_root, "test_config.yaml")
    csv_path = os.path.join(tmp_root, "results.csv")

    ok = True

    try:
        # ---- Setup ----
        os.makedirs(ckpt_dir)
        create_fake_hf_checkpoints(ckpt_dir, num_steps=3)
        write_test_config(config_path, ckpt_dir)

        # ---- Step 1: Extract containers ----
        step1_ok = run_script(
            os.path.join(project_root, "extract_containers.py"),
            ["--config", config_path, "--output_dir", container_dir],
            "Step 1: Extract",
        )
        ok = ok and step1_ok

        if step1_ok:
            # Verify container files were created
            container_files = []
            for root, dirs, files in os.walk(container_dir):
                for f in files:
                    if f.endswith(".safetensors"):
                        container_files.append(os.path.join(root, f))

            print(f"\n[verify] Found {len(container_files)} container file(s):")
            for cf in container_files:
                size_kb = os.path.getsize(cf) / 1024
                print(f"  {os.path.relpath(cf, tmp_root)}  ({size_kb:.1f} KB)")

            if len(container_files) == 0:
                print("[verify] FAILED — no container files created!")
                ok = False

        # ---- Step 2: Compression test ----
        if step1_ok:
            step2_ok = run_script(
                os.path.join(project_root, "run_compression_test.py"),
                [
                    "--config", config_path,
                    "--input_dir", container_dir,
                    "--output_csv", csv_path,
                ],
                "Step 2: Compress",
            )
            ok = ok and step2_ok

            if step2_ok and os.path.exists(csv_path):
                print(f"\n[verify] CSV output ({os.path.getsize(csv_path)} bytes):")
                with open(csv_path, "r") as f:
                    for line in f:
                        print(f"  {line.rstrip()}")
            elif step2_ok:
                print("[verify] WARNING — CSV file was not created")
                ok = False

    finally:
        # Cleanup
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\n[cleanup] Removed {tmp_root}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    if ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
