# Checkpoint Compression Toolkit

A modular, extensible toolkit for extracting tensors from large model checkpoints across training steps and benchmarking various compression methods on them.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
  - [Step 0: Convert to SafeTensors](#step-0-convert-to-safetensors)
  - [Step 1: Extract Containers](#step-1-extract-containers)
  - [Step 2: Run Compression Tests](#step-2-run-compression-tests)
- [Configuration Reference](#configuration-reference)
  - [YAML Config Structure](#yaml-config-structure)
  - [Example: HuggingFace (Amber)](#example-huggingface-amber)
  - [Example: Megatron (Flame-MoE)](#example-megatron-flame-moe)
- [Architecture](#architecture)
  - [Checkpoint Format Adapters](#checkpoint-format-adapters)
  - [Compression Method Plugins](#compression-method-plugins)
  - [Registry Pattern](#registry-pattern)
- [Extending the Toolkit](#extending-the-toolkit)
  - [Adding a New Checkpoint Format](#adding-a-new-checkpoint-format)
  - [Adding a New Compression Method](#adding-a-new-compression-method)
- [Built-in Components](#built-in-components)
  - [Checkpoint Adapters](#checkpoint-adapters)
  - [Compression Methods](#compression-methods)
- [Output Format](#output-format)
- [Advanced Usage](#advanced-usage)
  - [Resuming Interrupted Runs](#resuming-interrupted-runs)
  - [Skipping Rearrangement](#skipping-rearrangement)
  - [Selective Conversion](#selective-conversion)
- [Troubleshooting](#troubleshooting)

---

## Overview

When training large language models, checkpoints are saved periodically. This toolkit addresses the question: **How well can we compress checkpoint data, and does rearranging the data layout improve compressibility?**

The pipeline works in three stages:

1. **(Optional) Convert** raw checkpoints to SafeTensors format
2. **Extract** the same tensor (e.g., `model.layers.0.self_attn.q_proj.weight`) across all training steps into a single "container" file with structure `{step_0: tensor, step_1: tensor, ...}`
3. **Compress** each container using multiple methods, comparing the original `[Steps, Elements]` layout against a transposed `[Elements, Steps]` layout

The toolkit is designed to be **format-agnostic** and **method-agnostic** through a plugin architecture.

---

## Project Structure

```
checkstore_compression/
├── config/                              # YAML configuration files (one per model)
│   ├── amber.yaml                       #   HuggingFace Trainer format
│   └── flame_moe.yaml                   #   Megatron format
│
├── ckpt_formats/                        # Checkpoint format adapter plugins
│   ├── __init__.py                      #   Package init, re-exports
│   ├── base.py                          #   Abstract base class: CheckpointAdapter
│   ├── registry.py                      #   Adapter registry (register/get)
│   ├── huggingface.py                   #   HuggingFace adapter implementation
│   └── megatron.py                      #   Megatron adapter implementation
│
├── compression/                         # Compression method plugins
│   ├── __init__.py                      #   Package init, re-exports
│   ├── base.py                          #   Abstract base class: Compressor + CompressionResult
│   ├── registry.py                      #   Compressor registry (register/get)
│   ├── zstd_compressor.py               #   Plain ZSTD compression
│   ├── zstd_bytegrouping.py             #   ZSTD with byte-grouping pre-processing
│   └── zipnn_compressor.py              #   ZipNN (external script wrapper)
│
├── convert_to_safetensors.py            # Step 0: Format conversion entry point
├── extract_containers.py                # Step 1: Tensor extraction entry point
└── run_compression_test.py              # Step 2: Compression benchmark entry point
```

---

## Installation

### Dependencies

```bash
pip install torch safetensors numpy pandas pyyaml zstandard tqdm transformers
```

- **torch**: Tensor operations and dtype support
- **safetensors**: Fast, safe tensor serialization format
- **numpy**: Byte-level manipulation for byte-grouping
- **pandas**: CSV result aggregation
- **pyyaml**: Configuration file parsing
- **zstandard**: ZSTD compression library (Python bindings)
- **tqdm**: Progress bars
- **transformers**: Only required for Step 0 (checkpoint conversion)

### Optional

- **ZipNN**: If you want to use the ZipNN compression method, install it separately and point to its script in the config. See [ZipNN GitHub](https://github.com/zipnn/zipnn).

---

## Quick Start

```bash
# 1. (Optional) Convert checkpoints to safetensors
python convert_to_safetensors.py \
    --src_dir /path/to/raw_checkpoints \
    --dst_dir /path/to/converted_checkpoints \
    --pattern "ckpt_*" \
    --step_regex "ckpt_(\d+)"

# 2. Extract containers
python extract_containers.py \
    --config config/flame_moe.yaml \
    --output_dir ./extracted_containers

# 3. Run compression tests
python run_compression_test.py \
    --config config/flame_moe.yaml \
    --input_dir ./extracted_containers \
    --output_csv compression_results.csv
```

---

## Pipeline Overview

### Step 0: Convert to SafeTensors

**Script**: `convert_to_safetensors.py`

**Purpose**: Convert model checkpoints from any format supported by HuggingFace Transformers (PyTorch `.bin`, TensorFlow, Flax, etc.) into SafeTensors format. This step is only needed if your checkpoints are not already in SafeTensors.

**How it works**:
1. Discovers checkpoint directories matching a glob pattern
2. Extracts step numbers using a regex
3. Loads each checkpoint via `AutoModelForCausalLM.from_pretrained()`
4. Saves with `safe_serialization=True` and controllable shard size
5. Skips already-converted checkpoints automatically

**CLI Arguments**:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--src_dir` | Yes | - | Directory containing original checkpoint subdirectories |
| `--dst_dir` | Yes | - | Output directory for converted checkpoints |
| `--pattern` | No | `ckpt_*` | Glob pattern to match checkpoint directories |
| `--step_regex` | No | `ckpt_(\d+)` | Regex with one capture group for the step number |
| `--max_shard_size` | No | `5GB` | Maximum size per shard file |
| `--num_latest` | No | `0` | Only convert the N most recent checkpoints (0 = all) |
| `--dtype` | No | `float32` | Torch dtype: `float32`, `float16`, or `bfloat16` |

**Example**:

```bash
# Convert the last 10 Amber checkpoints
python convert_to_safetensors.py \
    --src_dir amber_ckpts \
    --dst_dir standardized_amber_ckpts \
    --pattern "ckpt_*" \
    --step_regex "ckpt_(\d+)" \
    --num_latest 10 \
    --max_shard_size 5GB
```

---

### Step 1: Extract Containers

**Script**: `extract_containers.py`

**Purpose**: For every parameter in the model (and optionally optimizer states), collect the same tensor across all training checkpoints and save it into a single SafeTensors "container" file.

**How it works**:
1. Reads the YAML config to determine checkpoint format and tensor types
2. Instantiates the appropriate `CheckpointAdapter` via the registry
3. Discovers all checkpoints and enumerates all parameter names
4. For each `(parameter, tensor_type)` pair:
   - Iterates over all checkpoints
   - Loads the tensor using the adapter
   - Saves all steps into one SafeTensors file with keys `step_0`, `step_1`, ...
5. Skips containers that already exist (for resumability)

**Output structure**:

```
extracted_containers/
├── model_layers_0_self_attn_q_proj_weight/
│   ├── weight.safetensors          # model weight across steps
│   ├── momentum.safetensors        # optimizer exp_avg across steps
│   ├── variance.safetensors        # optimizer exp_avg_sq across steps
│   └── master_weight.safetensors   # optimizer fp32_param across steps
├── model_layers_0_self_attn_k_proj_weight/
│   ├── weight.safetensors
│   └── ...
└── ...
```

Each `.safetensors` file contains:
```
{
    "step_100": tensor([...]),   # shape matches the parameter shape
    "step_200": tensor([...]),
    "step_300": tensor([...]),
    ...
}
```

**CLI Arguments**:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--config` | Yes | - | Path to YAML config file |
| `--output_dir` | No | `./extracted_containers` | Directory for output containers |

**Example**:

```bash
python extract_containers.py \
    --config config/flame_moe.yaml \
    --output_dir /data/flame_moe_containers
```

---

### Step 2: Run Compression Tests

**Script**: `run_compression_test.py`

**Purpose**: Benchmark compression methods on extracted container files, comparing the original layout (`[Steps, *Shape]`) against a rearranged layout (`[Elements, Steps]`).

**How it works**:
1. Reads compression method configs from YAML and instantiates compressor plugins
2. Discovers all `.safetensors` files in the input directory
3. For each file:
   - **Original layout test**: Runs all compressors on the file as-is
   - **Rearranged layout test** (optional):
     - Loads all `step_*` tensors, stacks them into `[Steps, Elements]`
     - Transposes to `[Elements, Steps]` (groups same-element values across steps)
     - Saves as a temporary file, runs all compressors, then deletes the temp file
4. Collects all results into a CSV with columns like `Orig_ZSTD`, `Rearranged_ZSTD+ByteGrouping`, etc.

**Rearrangement intuition**: When training a neural network, weights change gradually between steps. By transposing so that the same weight element across steps is contiguous in memory, delta-like patterns emerge that compress much better.

```
Original layout:      [step_0_all_weights, step_1_all_weights, ...]
Rearranged layout:    [weight_0_all_steps, weight_1_all_steps, ...]
```

**CLI Arguments**:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--config` | Yes | - | Path to YAML config file |
| `--input_dir` | Yes | - | Directory containing extracted containers |
| `--output_csv` | No | `compression_results.csv` | Output CSV path |
| `--skip_rearrange` | No | `false` | Skip the rearrangement step (flag) |

**Example**:

```bash
python run_compression_test.py \
    --config config/flame_moe.yaml \
    --input_dir /data/flame_moe_containers \
    --output_csv /data/results.csv
```

---

## Configuration Reference

### YAML Config Structure

A configuration file has the following top-level sections:

```yaml
# ---- Checkpoint discovery ----
checkpoint:
  base_dir: "/path/to/checkpoints"       # Root directory
  pattern: "iter_*"                       # Glob pattern for subdirectories
  step_regex: "iter_(\\d+)"              # Regex to extract step number (must have 1 capture group)

# ---- Format selection ----
format: megatron                          # Adapter name: "megatron" | "huggingface" | ...

# ---- Tensor types to extract ----
tensor_types:
  <type_name>:                            # Arbitrary name (e.g., "weight", "momentum")
    source: model | optimizer             # Which index file to look up
    key_template: "..."                   # Python format string with {param_name} and optional {chained_prefix}

# ---- Format-specific settings (optional) ----
megatron:                                 # Only used by the megatron adapter
  chained_prefix_default: "chained_0"
  chained_prefix_expert: "chained_1"
  chained_test_template: "chained_1.optimizer.state.exp_avg.{param_name}"

# ---- Compression methods ----
compression:
  methods:
    - name: zstd                          # Compressor registry name
      level: 3                            # Compressor-specific kwargs
    - name: zstd_bytegrouping
      level: 3
    - name: zipnn
      script_path: "/path/to/zipnn_compress_safetensors.py"
```

### Example: HuggingFace (Amber)

```yaml
# config/amber.yaml
checkpoint:
  base_dir: "amber_ckpts"
  pattern: "ckpt_*"
  step_regex: "ckpt_(\\d+)"

format: huggingface

tensor_types:
  weight:
    source: model
    key_template: "{param_name}"

compression:
  methods:
    - name: zstd
      level: 3
    - name: zstd_bytegrouping
      level: 3
```

This config:
- Looks for directories like `amber_ckpts/ckpt_100/`, `amber_ckpts/ckpt_200/`
- Only extracts model weights (no optimizer states, since HF Trainer doesn't always save them in SafeTensors)
- Tests ZSTD and ZSTD+ByteGrouping compression

### Example: Megatron (Flame-MoE)

```yaml
# config/flame_moe.yaml
checkpoint:
  base_dir: "/mnt/sda1/yxz/Flame-moe/converted_checkpoints"
  pattern: "iter_*"
  step_regex: "iter_(\\d+)"

format: megatron

tensor_types:
  weight:
    source: model
    key_template: "{param_name}"
  momentum:
    source: optimizer
    key_template: "{chained_prefix}.optimizer.state.exp_avg.{param_name}"
  variance:
    source: optimizer
    key_template: "{chained_prefix}.optimizer.state.exp_avg_sq.{param_name}"
  master_weight:
    source: optimizer
    key_template: "{chained_prefix}.optimizer.state.fp32_param.{param_name}"

megatron:
  chained_prefix_default: "chained_0"
  chained_prefix_expert: "chained_1"
  chained_test_template: "chained_1.optimizer.state.exp_avg.{param_name}"

compression:
  methods:
    - name: zstd
      level: 3
    - name: zstd_bytegrouping
      level: 3
    - name: zipnn
      script_path: "/mnt/sda1/yxz/zipnn/scripts/zipnn_compress_safetensors.py"
```

This config:
- Looks for directories like `converted_checkpoints/iter_00001/`
- Extracts 4 tensor types per parameter: weight, momentum, variance, master_weight
- The `{chained_prefix}` is auto-detected per parameter (expert layers use `chained_1`, others use `chained_0`)
- Tests 3 compression methods

---

## Architecture

### Checkpoint Format Adapters

The adapter pattern decouples the extraction pipeline from checkpoint format specifics.

```
CheckpointAdapter (Abstract Base Class)
├── discover_checkpoints() -> [(step, path), ...]
├── get_parameter_names() -> [str, ...]
├── get_tensor_types() -> [str, ...]
├── get_tensor_key(param_name, tensor_type) -> str | None
└── load_tensor(ckpt_path, param_name, tensor_type) -> Tensor | None
```

**Key design decisions**:
- **Lazy loading**: Index files (weight maps) are loaded on first access and cached
- **Format-specific logic is encapsulated**: The Megatron adapter handles `chained_prefix` detection; the HF adapter handles single-file vs. sharded models
- **Config-driven**: Tensor type definitions (key templates) come from YAML, not hardcoded

### Compression Method Plugins

Each compression method implements a simple interface:

```
Compressor (Abstract Base Class)
├── name: str                    # Display name
└── compress(filepath) -> CompressionResult | None

CompressionResult (Dataclass)
├── original_size: int
├── compressed_size: int
└── ratio: float (property)      # compressed_size / original_size
```

### Registry Pattern

Both adapters and compressors use the same registration mechanism:

```python
# Registration (in the module file)
@register_adapter("megatron")
class MegatronAdapter(CheckpointAdapter):
    ...

@register_compressor("zstd")
class ZstdCompressor(Compressor):
    ...

# Lookup (in the pipeline scripts)
adapter = get_adapter("megatron", config)
compressor = get_compressor("zstd", level=3)
```

Modules are registered at import time via decorators. The entry point scripts explicitly import all adapter/compressor modules to trigger registration.

---

## Extending the Toolkit

### Adding a New Checkpoint Format

1. Create `ckpt_formats/my_format.py`:

```python
from .base import CheckpointAdapter
from .registry import register_adapter

@register_adapter("my_format")
class MyFormatAdapter(CheckpointAdapter):
    def __init__(self, config: dict):
        super().__init__(config)
        # Read config["checkpoint"] and any format-specific sections

    def discover_checkpoints(self):
        # Return [(step_number, path), ...] sorted by step
        ...

    def get_parameter_names(self):
        # Return list of all parameter names from the first checkpoint
        ...

    def get_tensor_types(self):
        # Return list of tensor type names from config
        return list(self.config.get("tensor_types", {}).keys())

    def get_tensor_key(self, param_name, tensor_type):
        # Map (param_name, tensor_type) -> actual key in checkpoint file
        ...

    def load_tensor(self, ckpt_path, param_name, tensor_type):
        # Load and return a single tensor, or None
        ...
```

2. Register the import in `extract_containers.py`:

```python
import ckpt_formats.my_format  # noqa: F401
```

3. Create a config file `config/my_model.yaml`:

```yaml
checkpoint:
  base_dir: "/path/to/checkpoints"
  pattern: "step_*"
  step_regex: "step_(\\d+)"

format: my_format

tensor_types:
  weight:
    source: model
    key_template: "{param_name}"
```

### Adding a New Compression Method

1. Create `compression/my_compressor.py`:

```python
from .base import Compressor, CompressionResult
from .registry import register_compressor

@register_compressor("my_method")
class MyCompressor(Compressor):
    name = "MyMethod"

    def __init__(self, my_param: int = 5, **kwargs):
        self.my_param = my_param

    def compress(self, filepath):
        # Read the file, compress it, return CompressionResult
        with open(filepath, "rb") as f:
            data = f.read()
        original_size = len(data)

        # ... your compression logic ...
        compressed_size = ...

        return CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size
        )
```

2. Register the import in `run_compression_test.py`:

```python
import compression.my_compressor  # noqa: F401
```

3. Add to your YAML config:

```yaml
compression:
  methods:
    - name: my_method
      my_param: 10
```

That's it. The `**kwargs` in `__init__` ensures unknown config keys don't cause errors.

---

## Built-in Components

### Checkpoint Adapters

| Name | Registry Key | Description |
|---|---|---|
| `HuggingFaceAdapter` | `huggingface` | Standard HF Transformers checkpoints. Supports single-file and sharded SafeTensors. Model weights only. |
| `MegatronAdapter` | `megatron` | Megatron-LM style checkpoints with separate model and optimizer index files. Supports `chained_prefix` auto-detection for MoE models. |

### Compression Methods

| Name | Registry Key | Parameters | Description |
|---|---|---|---|
| `ZstdCompressor` | `zstd` | `level` (default: 3) | Plain ZSTD compression on the raw file bytes |
| `ZstdByteGroupingCompressor` | `zstd_bytegrouping` | `level` (default: 3) | Reorders tensor bytes by significance (e.g., all MSBs together, all LSBs together) before ZSTD compression. Preserves the SafeTensors header. |
| `ZipNNCompressor` | `zipnn` | `script_path` | Wraps an external ZipNN script. Parses compression ratio from stdout. Automatically cleans up generated files. |

#### Byte Grouping Explained

For a tensor with `float16` elements (2 bytes each), byte grouping transforms:

```
Original:  [A0_hi, A0_lo, A1_hi, A1_lo, A2_hi, A2_lo, ...]
Grouped:   [A0_hi, A1_hi, A2_hi, ..., A0_lo, A1_lo, A2_lo, ...]
```

This groups bytes of the same significance together, which often reveals patterns (e.g., many weights sharing similar exponent bytes) that compress much better.

---

## Output Format

The output CSV from `run_compression_test.py` has one row per container file:

| Column | Description |
|---|---|
| `File` | Absolute path to the container safetensors file |
| `Orig_ZSTD` | Compression ratio (compressed/original) for original layout with ZSTD |
| `Orig_ZSTD+ByteGrouping` | Compression ratio for original layout with ZSTD+ByteGrouping |
| `Orig_ZipNN` | Compression ratio for original layout with ZipNN |
| `Rearranged_ZSTD` | Compression ratio for rearranged (Elements x Steps) layout with ZSTD |
| `Rearranged_ZSTD+ByteGrouping` | Compression ratio for rearranged layout with ZSTD+ByteGrouping |
| `Rearranged_ZipNN` | Compression ratio for rearranged layout with ZipNN |

**Compression ratio** = `compressed_size / original_size` (lower is better; 0.5 means 50% of original size).

---

## Benchmark Results

Tested on **Flame-MoE** (Megatron format, bfloat16, 10 checkpoints across ~11k training steps). 12 representative parameters covering all layer types:

### Large Tensors (Linear / Expert / Embedding weights)

| Parameter | Size | Orig ZSTD | Orig BG | Orig ZipNN | Rearr ZSTD | Rearr BG | Rearr ZipNN |
|---|---|---|---|---|---|---|---|
| `L0.self_attention.linear_qkv.weight` | 240 MB | 0.779 | 0.711 | 0.663 | 0.743 | **0.633** | 0.663 |
| `L0.self_attention.linear_proj.weight` | 80 MB | 0.783 | 0.713 | 0.664 | 0.775 | **0.667** | 0.666 |
| `L0.mlp.linear_fc1.weight` | 855 MB | 0.782 | 0.713 | 0.663 | 0.727 | **0.620** | 0.663 |
| `L0.mlp.linear_fc2.weight` | 428 MB | 0.781 | 0.712 | 0.664 | 0.763 | **0.650** | 0.665 |
| `L1.mlp.experts.experts.linear_fc1.weight` | 7040 MB | 0.782 | 0.714 | 0.663 | 0.734 | **0.626** | 0.663 |
| `L1.mlp.experts.experts.linear_fc2.weight` | 3520 MB | 0.782 | 0.714 | 0.667 | 0.773 | **0.658** | 0.670 |
| `L1.mlp.router.weight` | 2.5 MB | 0.780 | 0.694 | 0.661 | 0.743 | **0.629** | 0.662 |
| `embedding.word_embeddings.weight` | 1965 MB | 0.783 | 0.714 | 0.662 | 0.747 | **0.635** | 0.662 |
| `output_layer.weight` | 1965 MB | 0.783 | 0.713 | 0.662 | 0.756 | **0.642** | 0.664 |

### Small Tensors (LayerNorm weights)

| Parameter | Size | Orig ZSTD | Orig BG | Orig ZipNN | Rearr ZSTD | Rearr BG | Rearr ZipNN |
|---|---|---|---|---|---|---|---|
| `final_layernorm.weight` | 41 KB | 0.353 | 0.259 | 0.344 | 0.244 | **0.197** | 0.330 |
| `L0.linear_qkv.layer_norm_weight` | 41 KB | 0.439 | 0.385 | 0.345 | 0.368 | **0.340** | 0.386 |
| `L1.pre_mlp_layernorm.weight` | 41 KB | 0.375 | 0.311 | 0.343 | 0.327 | **0.283** | 0.404 |

*BG = ZSTD+ByteGrouping, Rearr = Rearranged (Elements x Steps) layout. Bold = best ratio per row.*

### Key Findings

1. **ZSTD+ByteGrouping + Rearranged layout** is the best universal combination, achieving **0.62-0.67** on large tensors and **0.20-0.34** on small tensors
2. **ZipNN** performs well on original layout (~0.66 on large tensors) but is insensitive to rearrangement — it neither benefits from nor is harmed by the transposed layout
3. **Rearrangement** consistently improves ZSTD and ZSTD+ByteGrouping because adjacent training steps produce nearly identical weight values, and the transposed layout groups these similar values contiguously
4. **Small tensors** (LayerNorm) compress much better than large tensors across all methods, likely due to lower entropy in normalization parameters

---

## Testing

The project includes three test scripts:

### `test_pipeline.py` — End-to-end test (synthetic data)

```bash
python test_pipeline.py
```

Creates fake HuggingFace checkpoints in a temp directory, runs the full pipeline (extract + compress), and verifies output correctness. No real data needed.

### `test_real_data.py` — Quick smoke test (real data)

```bash
python test_real_data.py
```

Uses the first 2 parameters from Flame-MoE checkpoints with all 3 compression methods. Fast (~10 seconds).

### `test_flame_moe_full.py` — Comprehensive benchmark (real data)

```bash
python test_flame_moe_full.py
```

Tests 12 representative parameters (one per layer type) with all 3 compression methods on both layouts. Produces the benchmark results table above. Takes ~30 minutes due to large tensor sizes (up to 7 GB per container).

---

## Advanced Usage

### Resuming Interrupted Runs

Both `extract_containers.py` and `run_compression_test.py` support automatic resumption:

- **extract_containers.py**: Skips container files that already exist and have non-zero size
- **run_compression_test.py**: Loads the existing CSV and skips files already present in it, then appends new results

This means you can safely `Ctrl+C` and re-run the same command.

### Skipping Rearrangement

If you only want to test compression on the original layout:

```bash
python run_compression_test.py \
    --config config/flame_moe.yaml \
    --input_dir ./containers \
    --output_csv results.csv \
    --skip_rearrange
```

### Selective Conversion

Convert only the 5 most recent checkpoints:

```bash
python convert_to_safetensors.py \
    --src_dir amber_ckpts \
    --dst_dir standardized \
    --num_latest 5
```

---

## Troubleshooting

### "No checkpoints found"

- Verify `checkpoint.base_dir` exists and contains subdirectories
- Verify `checkpoint.pattern` matches the directory names (test with `ls /path/to/base_dir/pattern`)
- Verify `checkpoint.step_regex` correctly captures the step number:
  ```python
  import re
  re.search(r"iter_(\d+)", "iter_00001").group(1)  # Should return "00001" -> 1
  ```

### "Unknown checkpoint format 'xxx'"

- Ensure the adapter module is imported in `extract_containers.py`:
  ```python
  import ckpt_formats.xxx  # noqa: F401
  ```

### "Unknown compressor 'xxx'"

- Ensure the compressor module is imported in `run_compression_test.py`:
  ```python
  import compression.xxx  # noqa: F401
  ```

### ZipNN returns None

- Check that `script_path` in the YAML config points to a valid Python script
- Ensure ZipNN is installed and the script is executable
- Check stdout for "ratio is X.XXX" pattern — the parser relies on this

### Memory issues with large models

- The extraction script loads one tensor at a time per checkpoint (not the whole model)
- For very large models, ensure you have enough RAM to hold `num_checkpoints * single_tensor_size`
- The compression test rearrangement step loads all steps of one container into memory simultaneously

### Byte grouping fails for custom dtypes

- The byte grouping compressor uses `tensor.view(torch.uint8)` which works for all standard PyTorch dtypes (float16, bfloat16, float32, float64, int8, int16, int32, int64)
- For custom or quantized dtypes, it falls back to `tensor.cpu().numpy().tobytes()`
