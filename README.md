# Checkpoint 压缩测试工具包

一套模块化、可扩展的工具，用于从大模型训练 checkpoint 中提取 tensor，并对多种压缩方法进行基准测试。

## 目录

- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [环境安装](#环境安装)
- [快速上手](#快速上手)
- [流水线详解](#流水线详解)
  - [Step 0: 格式转换](#step-0-格式转换)
  - [Step 1: 提取 Container](#step-1-提取-container)
  - [Step 2: 压缩测试](#step-2-压缩测试)
- [配置文件参考](#配置文件参考)
  - [YAML 配置结构](#yaml-配置结构)
  - [示例：HuggingFace (Amber)](#示例huggingface-amber)
  - [示例：Megatron (Flame-MoE)](#示例megatron-flame-moe)
- [架构设计](#架构设计)
  - [Checkpoint 格式适配器](#checkpoint-格式适配器)
  - [压缩方法插件](#压缩方法插件)
  - [注册器模式](#注册器模式)
- [如何扩展](#如何扩展)
  - [添加新的 Checkpoint 格式](#添加新的-checkpoint-格式)
  - [添加新的压缩方法](#添加新的压缩方法)
- [内置组件说明](#内置组件说明)
  - [Checkpoint 适配器](#checkpoint-适配器)
  - [压缩方法](#压缩方法)
- [输出格式](#输出格式)
- [进阶用法](#进阶用法)
- [常见问题排查](#常见问题排查)

---

## 项目概述

大模型训练过程中会周期性保存 checkpoint。本工具回答的核心问题是：**checkpoint 数据的可压缩性如何？重新排列数据布局能否提升压缩率？**

流水线分为三个阶段：

1. **（可选）格式转换** — 将原始 checkpoint 转换为 SafeTensors 格式
2. **提取 Container** — 将同一个 tensor（如 `model.layers.0.self_attn.q_proj.weight`）在所有训练步骤中的值收集到一个 "container" 文件中，结构为 `{step_0: tensor, step_1: tensor, ...}`
3. **压缩测试** — 对每个 container 使用多种压缩方法进行测试，同时对比原始布局 `[Steps, Elements]` 和转置布局 `[Elements, Steps]` 的压缩效果

工具通过**插件架构**实现了对 checkpoint 格式和压缩方法的完全解耦。

### 为什么要重排数据布局？

原始布局中，每个 step 的完整 tensor 连续存储：

```
Step1: [e1, e2, e3, ...]  Step2: [e1, e2, e3, ...]  Step3: [e1, e2, e3, ...]
```

重排后，同一元素在不同 step 的值连续存储：

```
Element1: [s1, s2, s3, ...]  Element2: [s1, s2, s3, ...]  Element3: [s1, s2, s3, ...]
```

由于相邻训练步骤间同一参数的变化非常微小，重排后相邻字节之间的差异更小，更有利于压缩算法发现冗余。

---

## 项目结构

```
checkstore_compression/
├── config/                              # YAML 配置文件（每个模型一个）
│   ├── amber.yaml                       #   HuggingFace Trainer 格式
│   └── flame_moe.yaml                   #   Megatron 格式
│
├── ckpt_formats/                        # Checkpoint 格式适配器插件
│   ├── __init__.py                      #   包初始化，导出公共接口
│   ├── base.py                          #   抽象基类：CheckpointAdapter
│   ├── registry.py                      #   适配器注册器（注册/获取）
│   ├── huggingface.py                   #   HuggingFace 适配器实现
│   └── megatron.py                      #   Megatron 适配器实现
│
├── compression/                         # 压缩方法插件
│   ├── __init__.py                      #   包初始化，导出公共接口
│   ├── base.py                          #   抽象基类：Compressor + CompressionResult
│   ├── registry.py                      #   压缩器注册器（注册/获取）
│   ├── zstd_compressor.py               #   纯 ZSTD 压缩
│   ├── zstd_bytegrouping.py             #   ZSTD + 字节分组预处理
│   └── zipnn_compressor.py              #   ZipNN（外部脚本封装）
│
├── convert_to_safetensors.py            # Step 0：格式转换入口
├── extract_containers.py                # Step 1：提取 Container 入口
├── run_compression_test.py              # Step 2：压缩测试入口
├── test_pipeline.py                     # 端到端测试（模拟数据）
└── test_real_data.py                    # 真实数据冒烟测试
```

---

## 环境安装

### 必需依赖

```bash
pip install torch safetensors numpy pandas pyyaml zstandard tqdm transformers
```

| 依赖 | 用途 |
|---|---|
| `torch` | Tensor 操作与数据类型支持 |
| `safetensors` | 快速、安全的 tensor 序列化格式 |
| `numpy` | 字节级操作（字节分组） |
| `pandas` | CSV 结果汇总 |
| `pyyaml` | 解析 YAML 配置文件 |
| `zstandard` | ZSTD 压缩库（Python 绑定） |
| `tqdm` | 进度条 |
| `transformers` | 仅 Step 0（格式转换）需要 |

### 可选依赖

- **ZipNN**：如需使用 ZipNN 压缩方法，需单独安装并在配置中指定脚本路径。

---

## 快速上手

```bash
# 1.（可选）将 checkpoint 转换为 safetensors 格式
python convert_to_safetensors.py \
    --src_dir /path/to/raw_checkpoints \
    --dst_dir /path/to/converted_checkpoints \
    --pattern "ckpt_*" \
    --step_regex "ckpt_(\d+)"

# 2. 提取 container
python extract_containers.py \
    --config config/flame_moe.yaml \
    --output_dir ./extracted_containers

# 3. 运行压缩测试
python run_compression_test.py \
    --config config/flame_moe.yaml \
    --input_dir ./extracted_containers \
    --output_csv compression_results.csv
```

---

## 流水线详解

### Step 0: 格式转换

**脚本**：`convert_to_safetensors.py`

**用途**：将 HuggingFace Transformers 支持的任意格式（PyTorch `.bin`、TensorFlow、Flax 等）转换为 SafeTensors 格式。仅在 checkpoint 尚未使用 SafeTensors 格式时需要。

**工作流程**：
1. 根据 glob 模式发现 checkpoint 目录
2. 用正则表达式从目录名中提取 step 编号
3. 通过 `AutoModelForCausalLM.from_pretrained()` 加载每个 checkpoint
4. 以 `safe_serialization=True` 保存，支持控制分片大小
5. 自动跳过已转换的 checkpoint

**命令行参数**：

| 参数 | 必需 | 默认值 | 说明 |
|---|---|---|---|
| `--src_dir` | 是 | — | 原始 checkpoint 所在目录 |
| `--dst_dir` | 是 | — | 转换后输出目录 |
| `--pattern` | 否 | `ckpt_*` | checkpoint 目录的 glob 匹配模式 |
| `--step_regex` | 否 | `ckpt_(\d+)` | 从目录名提取 step 编号的正则表达式 |
| `--max_shard_size` | 否 | `5GB` | 输出 safetensors 文件的最大分片大小 |
| `--num_latest` | 否 | `0` | 仅转换最近 N 个 checkpoint（0 = 全部） |
| `--dtype` | 否 | `float32` | 加载时使用的数据类型（`float32`/`float16`/`bfloat16`） |

**示例**：

```bash
# 转换 Amber 模型最近 10 个 checkpoint
python convert_to_safetensors.py \
    --src_dir amber_ckpts \
    --dst_dir standardized_amber_ckpts \
    --pattern "ckpt_*" \
    --step_regex "ckpt_(\d+)" \
    --num_latest 10 \
    --max_shard_size 5GB
```

---

### Step 1: 提取 Container

**脚本**：`extract_containers.py`

**用途**：遍历所有 checkpoint，将同一参数的同一类型 tensor 跨所有训练步骤收集到一个 safetensors 文件中。

**工作流程**：
1. 读取 YAML 配置，创建对应格式的适配器
2. 发现所有 checkpoint 并按 step 排序
3. 获取参数列表和 tensor 类型列表
4. 对每个 (参数, tensor类型) 组合：
   - 遍历所有 checkpoint，加载对应 tensor
   - 保存为 `{step_N: tensor, ...}` 格式的 safetensors 文件
5. 已存在的文件自动跳过（支持断点续跑）

**命令行参数**：

| 参数 | 必需 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | 是 | — | YAML 配置文件路径 |
| `--output_dir` | 否 | `./extracted_containers` | container 输出目录 |

**输出目录结构**：

```
extracted_containers/
├── decoder_final_layernorm_weight/
│   └── weight.safetensors              # 包含 step_1100, step_2200, ... 的 tensor
├── decoder_layers_0_mlp_linear_fc1_weight/
│   ├── weight.safetensors
│   ├── momentum.safetensors            # (Megatron 格式才有)
│   ├── variance.safetensors
│   └── master_weight.safetensors
└── ...
```

---

### Step 2: 压缩测试

**脚本**：`run_compression_test.py`

**用途**：对提取出的 container 文件进行多种压缩方法的基准测试，同时对比原始布局和重排布局的压缩效果。

**工作流程**：
1. 读取 YAML 配置，实例化所有配置的压缩器
2. 递归扫描输入目录中的所有 `.safetensors` 文件
3. 对每个文件：
   - 使用所有压缩方法测试原始布局
   - 将数据重排为 `[Elements, Steps]` 布局
   - 使用所有压缩方法测试重排布局
   - 清理临时重排文件
4. 将结果追加到 CSV 文件（支持断点续跑）

**命令行参数**：

| 参数 | 必需 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | 是 | — | YAML 配置文件路径 |
| `--input_dir` | 是 | — | container 文件所在目录 |
| `--output_csv` | 否 | `compression_results.csv` | 结果输出 CSV 路径 |
| `--skip_rearrange` | 否 | `false` | 跳过重排步骤，仅测试原始布局 |

**示例**：

```bash
# 完整测试
python run_compression_test.py \
    --config config/flame_moe.yaml \
    --input_dir ./extracted_containers \
    --output_csv results.csv

# 仅测试原始布局（跳过重排）
python run_compression_test.py \
    --config config/flame_moe.yaml \
    --input_dir ./extracted_containers \
    --output_csv results.csv \
    --skip_rearrange
```

---

## 配置文件参考

### YAML 配置结构

配置文件是整个工具的核心，它定义了 checkpoint 的位置、格式、要提取的 tensor 类型以及要使用的压缩方法。

```yaml
# ===== 必需字段 =====

checkpoint:
  base_dir: "/path/to/checkpoints"     # checkpoint 根目录
  pattern: "iter_*"                     # 匹配 checkpoint 子目录的 glob 模式
  step_regex: "iter_(\\d+)"            # 从目录名提取 step 编号的正则（注意 YAML 中反斜杠需要转义）

format: megatron                        # 适配器名称（megatron / huggingface）

tensor_types:                           # 要提取的 tensor 类型定义
  weight:                               #   类型名（自定义，用于输出文件命名）
    source: model                       #   数据来源：model 或 optimizer
    key_template: "{param_name}"        #   在 checkpoint 文件中的 key 模板

# ===== 可选字段 =====

compression:                            # 压缩测试配置
  methods:
    - name: zstd                        #   压缩器注册名
      level: 3                          #   传递给压缩器的参数
    - name: zstd_bytegrouping
      level: 3
    - name: zipnn
      script_path: "/path/to/zipnn_compress_safetensors.py"

# ===== 格式特有字段 =====

megatron:                               # Megatron 格式专用配置
  chained_prefix_default: "chained_0"
  chained_prefix_expert: "chained_1"
  chained_test_template: "chained_1.optimizer.state.exp_avg.{param_name}"
```

### key_template 模板变量

模板中可使用以下变量：

| 变量 | 说明 | 示例 |
|---|---|---|
| `{param_name}` | 参数全名 | `decoder.layers.0.self_attention.linear_qkv.weight` |
| `{chained_prefix}` | Megatron 链式前缀（仅 Megatron 格式） | `chained_0` 或 `chained_1` |

### 示例：HuggingFace (Amber)

```yaml
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

### 示例：Megatron (Flame-MoE)

```yaml
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

---

## 架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      YAML 配置文件                           │
│  (checkpoint 路径、格式、tensor 类型、压缩方法)               │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│   ckpt_formats/          │  │   compression/               │
│                          │  │                              │
│  CheckpointAdapter (ABC) │  │  Compressor (ABC)            │
│    ├── MegatronAdapter   │  │    ├── ZstdCompressor        │
│    ├── HuggingFaceAdapter│  │    ├── ZstdByteGrouping      │
│    └── (自定义...)        │  │    ├── ZipNNCompressor       │
│                          │  │    └── (自定义...)            │
│  通过 @register_adapter  │  │  通过 @register_compressor   │
│  装饰器自动注册          │  │  装饰器自动注册              │
└──────────┬───────────────┘  └──────────┬───────────────────┘
           │                             │
           ▼                             ▼
┌────────────────────┐       ┌────────────────────────┐
│ extract_containers │       │ run_compression_test   │
│    .py             │──────▶│    .py                 │
│                    │       │                        │
│ 遍历 checkpoint    │       │ 对每个 container:      │
│ 收集同一 tensor    │       │   原始布局 → 压缩      │
│ 保存为 container   │       │   重排布局 → 压缩      │
│                    │       │   输出 CSV             │
└────────────────────┘       └────────────────────────┘
```

### Checkpoint 格式适配器

每种 checkpoint 格式实现 `CheckpointAdapter` 抽象基类：

```python
class CheckpointAdapter(ABC):
    def discover_checkpoints(self) -> List[Tuple[int, str]]:
        """返回 [(step编号, 路径), ...] 按 step 排序"""

    def get_parameter_names(self) -> List[str]:
        """返回所有参数名"""

    def get_tensor_types(self) -> List[str]:
        """返回支持的 tensor 类型名"""

    def get_tensor_key(self, param_name, tensor_type) -> Optional[str]:
        """给定参数名和类型，返回在文件中的实际 key"""

    def load_tensor(self, ckpt_path, param_name, tensor_type) -> Optional[Tensor]:
        """从 checkpoint 加载一个 tensor"""
```

适配器负责处理所有格式差异：
- **Megatron**：处理 model/optimizer 分离的 index 文件、chained prefix 逻辑
- **HuggingFace**：处理单文件/分片模型、自动检测 index 文件

### 压缩方法插件

每种压缩方法实现 `Compressor` 抽象基类：

```python
@dataclass
class CompressionResult:
    original_size: int       # 原始大小（字节）
    compressed_size: int     # 压缩后大小（字节）
    ratio: float             # 压缩比 = compressed / original（越小越好）

class Compressor(ABC):
    name: str                # 显示名称

    def compress(self, filepath: str) -> Optional[CompressionResult]:
        """压缩文件并返回结果，失败返回 None"""
```

### 注册器模式

适配器和压缩器都使用装饰器自动注册：

```python
# 注册
@register_adapter("megatron")
class MegatronAdapter(CheckpointAdapter): ...

@register_compressor("zstd")
class ZstdCompressor(Compressor): ...

# 使用
adapter = get_adapter("megatron", config)      # 按名称获取
compressor = get_compressor("zstd", level=3)   # 按名称获取，传入参数
```

配置文件中的 `format` 字段和 `compression.methods[].name` 字段对应注册名。

---

## 如何扩展

### 添加新的 Checkpoint 格式

以 DeepSpeed 为例：

**1. 创建适配器文件** `ckpt_formats/deepspeed.py`：

```python
from .base import CheckpointAdapter
from .registry import register_adapter

@register_adapter("deepspeed")
class DeepSpeedAdapter(CheckpointAdapter):
    def __init__(self, config: dict):
        super().__init__(config)
        ckpt_cfg = config["checkpoint"]
        self.base_dir = ckpt_cfg["base_dir"]
        self.pattern = ckpt_cfg.get("pattern", "global_step*")
        self.step_regex = ckpt_cfg.get("step_regex", r"global_step(\d+)")
        self.tensor_types_cfg = config.get("tensor_types", {})

    def discover_checkpoints(self):
        # ... 实现 checkpoint 发现逻辑
        pass

    def get_parameter_names(self):
        # ... 从 DeepSpeed 的 state dict 中获取参数名
        pass

    def get_tensor_types(self):
        return list(self.tensor_types_cfg.keys())

    def get_tensor_key(self, param_name, tensor_type):
        # ... 根据 DeepSpeed 的 key 命名规则返回
        pass

    def load_tensor(self, ckpt_path, param_name, tensor_type):
        # ... 从 DeepSpeed checkpoint 加载 tensor
        pass
```

**2. 在入口脚本中导入**（`extract_containers.py`）：

```python
import ckpt_formats.deepspeed  # noqa: F401
```

**3. 编写配置文件** `config/my_deepspeed_model.yaml`：

```yaml
checkpoint:
  base_dir: "/path/to/deepspeed_checkpoints"
  pattern: "global_step*"
  step_regex: "global_step(\\d+)"

format: deepspeed

tensor_types:
  weight:
    source: model
    key_template: "{param_name}"
```

### 添加新的压缩方法

以 LZ4 为例：

**1. 创建压缩器文件** `compression/lz4_compressor.py`：

```python
import lz4.frame
from .base import Compressor, CompressionResult
from .registry import register_compressor

@register_compressor("lz4")
class LZ4Compressor(Compressor):
    name = "LZ4"

    def __init__(self, **kwargs):
        pass

    def compress(self, filepath):
        with open(filepath, "rb") as f:
            data = f.read()
        compressed = lz4.frame.compress(data)
        return CompressionResult(
            original_size=len(data),
            compressed_size=len(compressed),
        )
```

**2. 在入口脚本中导入**（`run_compression_test.py`）：

```python
import compression.lz4_compressor  # noqa: F401
```

**3. 在配置文件中启用**：

```yaml
compression:
  methods:
    - name: lz4
    - name: zstd
      level: 3
```

---

## 内置组件说明

### Checkpoint 适配器

#### `megatron` — Megatron-LM / Flame-MoE

适用于 Megatron-LM 框架保存的 checkpoint，特点：
- 模型权重和优化器状态分别存储在不同的 safetensors 文件中
- 通过 `model.safetensors.index.json` 和 `optimizer.safetensors.index.json` 索引
- 优化器 key 带有 `chained_0` / `chained_1` 前缀（expert 层使用 `chained_1`）
- 支持 weight、momentum（exp_avg）、variance（exp_avg_sq）、master_weight（fp32_param）

#### `huggingface` — HuggingFace Trainer

适用于 HuggingFace Transformers 保存的 checkpoint，特点：
- 支持单文件 `model.safetensors` 和分片 `model-00001-of-NNNNN.safetensors`
- 自动检测 index 文件或单文件模式
- 仅支持模型权重（不含优化器状态）

### 压缩方法

#### `zstd` — 纯 ZSTD 压缩

直接对 safetensors 文件的原始字节进行 ZSTD 压缩。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `level` | `3` | ZSTD 压缩级别（1-22，越高压缩率越好但越慢） |

#### `zstd_bytegrouping` — ZSTD + 字节分组

先对 tensor 数据进行字节分组预处理，再用 ZSTD 压缩。

**字节分组原理**：对于多字节数据类型（如 float32 = 4 字节），将所有元素的第 1 个字节放在一起、第 2 个字节放在一起，以此类推。这样同一 significance 的字节聚集在一起，提高了数据的局部相似性。

```
原始 (3 个 float32):
  [B0 B1 B2 B3] [B0 B1 B2 B3] [B0 B1 B2 B3]

字节分组后:
  [B0 B0 B0] [B1 B1 B1] [B2 B2 B2] [B3 B3 B3]
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `level` | `3` | ZSTD 压缩级别 |

#### `zipnn` — ZipNN 压缩

调用外部 ZipNN 脚本进行压缩，通过解析脚本输出获取压缩比。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `script_path` | `""` | ZipNN 压缩脚本的绝对路径 |

---

## 输出格式

压缩测试的输出为 CSV 文件，每行对应一个 container 文件：

| 列名 | 说明 |
|---|---|
| `File` | container 文件的完整路径 |
| `Orig_ZSTD` | 原始布局下 ZSTD 的压缩比 |
| `Orig_ZSTD+ByteGrouping` | 原始布局下 ZSTD+字节分组的压缩比 |
| `Orig_ZipNN` | 原始布局下 ZipNN 的压缩比 |
| `Rearranged_ZSTD` | 重排布局下 ZSTD 的压缩比 |
| `Rearranged_ZSTD+ByteGrouping` | 重排布局下 ZSTD+字节分组的压缩比 |
| `Rearranged_ZipNN` | 重排布局下 ZipNN 的压缩比 |

**压缩比含义**：`compressed_size / original_size`，值越小表示压缩效果越好。例如 `0.35` 表示压缩后体积为原来的 35%。

### 真实数据基准测试结果

以 **Flame-MoE** 模型（Megatron 格式，bfloat16，10 个 checkpoint，训练步数 1100~11029）为例，测试了 12 个代表性参数：

#### 大 Tensor（线性层 / Expert / Embedding 权重）

| 参数 | 大小 | 原始 ZSTD | 原始 BG | 原始 ZipNN | 重排 ZSTD | 重排 BG | 重排 ZipNN |
|---|---|---|---|---|---|---|---|
| `L0.self_attention.linear_qkv.weight` | 240 MB | 0.779 | 0.711 | 0.663 | 0.743 | **0.633** | 0.663 |
| `L0.self_attention.linear_proj.weight` | 80 MB | 0.783 | 0.713 | 0.664 | 0.775 | **0.667** | 0.666 |
| `L0.mlp.linear_fc1.weight` | 855 MB | 0.782 | 0.713 | 0.663 | 0.727 | **0.620** | 0.663 |
| `L0.mlp.linear_fc2.weight` | 428 MB | 0.781 | 0.712 | 0.664 | 0.763 | **0.650** | 0.665 |
| `L1.experts.linear_fc1.weight` | 7040 MB | 0.782 | 0.714 | 0.663 | 0.734 | **0.626** | 0.663 |
| `L1.experts.linear_fc2.weight` | 3520 MB | 0.782 | 0.714 | 0.667 | 0.773 | **0.658** | 0.670 |
| `L1.mlp.router.weight` | 2.5 MB | 0.780 | 0.694 | 0.661 | 0.743 | **0.629** | 0.662 |
| `embedding.word_embeddings.weight` | 1965 MB | 0.783 | 0.714 | 0.662 | 0.747 | **0.635** | 0.662 |
| `output_layer.weight` | 1965 MB | 0.783 | 0.713 | 0.662 | 0.756 | **0.642** | 0.664 |

*BG = ZSTD+ByteGrouping，加粗 = 该行最佳压缩比*

#### 小 Tensor（LayerNorm 权重）

| 参数 | 大小 | 原始 ZSTD | 原始 BG | 原始 ZipNN | 重排 ZSTD | 重排 BG | 重排 ZipNN |
|---|---|---|---|---|---|---|---|
| `final_layernorm.weight` | 41 KB | 0.353 | 0.259 | 0.344 | 0.244 | **0.197** | 0.330 |
| `L0.linear_qkv.layer_norm_weight` | 41 KB | 0.439 | 0.385 | 0.345 | 0.368 | **0.340** | 0.386 |
| `L1.pre_mlp_layernorm.weight` | 41 KB | 0.375 | 0.311 | 0.343 | 0.327 | **0.283** | 0.404 |

#### 关键发现

1. **ZSTD+ByteGrouping + 重排布局** 是最佳通用组合，大 tensor 可达 **0.62~0.67**，小 tensor 可达 **0.20~0.34**
2. **ZipNN** 在原始布局上表现不错（大 tensor 稳定在 ~0.66），但对重排不敏感——重排后既不提升也不恶化
3. **重排布局** 一致提升 ZSTD 和 ZSTD+ByteGrouping 的效果，因为相邻训练步骤的同一权重值非常接近，转置后这些相似值被排列在一起
4. **小 tensor**（LayerNorm）在所有方法下都比大 tensor 压缩效果好，可能因为归一化参数的熵更低

| 场景 | 推荐方法 |
|---|---|
| 小 tensor | ZSTD+ByteGrouping + 重排（0.19~0.28） |
| 大 tensor，原始布局 | ZipNN（~0.66） |
| 大 tensor，重排布局 | ZSTD+ByteGrouping（0.62~0.65） |
| 通用最佳组合 | **ZSTD+ByteGrouping + 重排布局** |

---

## 进阶用法

### 断点续跑

**提取阶段**：已存在且非空的 container 文件会自动跳过。如果提取中途中断，重新运行即可从断点继续。

**压缩测试阶段**：已存在于 CSV 中的文件会自动跳过。新结果追加到现有 CSV 中。

### 跳过重排

如果只关心原始布局的压缩效果：

```bash
python run_compression_test.py \
    --config config/flame_moe.yaml \
    --input_dir ./containers \
    --output_csv results.csv \
    --skip_rearrange
```

### 选择性转换

只转换最近 5 个 checkpoint：

```bash
python convert_to_safetensors.py \
    --src_dir /path/to/ckpts \
    --dst_dir /path/to/output \
    --num_latest 5
```

使用 bfloat16 精度加载以节省内存：

```bash
python convert_to_safetensors.py \
    --src_dir /path/to/ckpts \
    --dst_dir /path/to/output \
    --dtype bfloat16
```

---

## 常见问题排查

### 找不到 checkpoint

- 确认 `checkpoint.base_dir` 路径存在且包含子目录
- 确认 `checkpoint.pattern` 能匹配到目录名（可用 `ls /path/to/base_dir/pattern` 测试）
- 确认 `checkpoint.step_regex` 能正确提取 step 编号：
  ```python
  import re
  re.search(r"iter_(\d+)", "iter_00001").group(1)  # 应返回 "00001" -> 1
  ```

### "Unknown checkpoint format 'xxx'"

确保适配器模块已在 `extract_containers.py` 中导入：
```python
import ckpt_formats.xxx  # noqa: F401
```

### "Unknown compressor 'xxx'"

确保压缩器模块已在 `run_compression_test.py` 中导入：
```python
import compression.xxx  # noqa: F401
```

### ZipNN 返回 None

- 检查 YAML 配置中 `script_path` 是否指向有效的 Python 脚本
- 确保 ZipNN 已安装且脚本可执行
- 检查脚本输出中是否包含 `ratio is X.XXX` 格式的文本（解析器依赖此格式）

### 大模型内存不足

- 提取脚本每次只加载一个 checkpoint 中的一个 tensor（不会加载整个模型）
- 对于超大模型，确保内存足够容纳 `checkpoint数量 × 单个tensor大小`
- 压缩测试的重排步骤会同时加载一个 container 的所有 step 到内存中

### 字节分组对自定义数据类型失败

- 字节分组使用 `tensor.view(torch.uint8)` 实现，支持所有标准 PyTorch 数据类型（float16、bfloat16、float32、float64、int8、int16、int32、int64）
- 对于自定义或量化数据类型，会自动回退到 `tensor.cpu().numpy().tobytes()`

### NumPy 版本兼容性警告

如果看到 `_ARRAY_API not found` 警告，这是 pandas/numexpr 与 NumPy 2.x 的兼容性问题，不影响功能。可通过以下方式解决：
```bash
pip install "numpy<2"
# 或升级 pandas
pip install --upgrade pandas
```

---

## 测试

项目包含三个测试脚本：

### `test_pipeline.py` — 端到端测试（模拟数据）

```bash
python test_pipeline.py
```

创建模拟的 HuggingFace checkpoint，运行完整流水线（提取 + 压缩），验证输出正确性。不依赖任何真实数据。

### `test_real_data.py` — 快速冒烟测试（真实数据）

```bash
python test_real_data.py
```

使用 Flame-MoE 真实 checkpoint 的前 2 个参数，配合 ZSTD、ZSTD+ByteGrouping、ZipNN 三种方法进行快速验证。约 10 秒完成。

### `test_flame_moe_full.py` — 全面基准测试（真实数据）

```bash
python test_flame_moe_full.py
```

选取 12 个代表性参数（每种层类型各一个），使用全部 3 种压缩方法分别在原始布局和重排布局上测试。输出上面"基准测试结果"章节中的完整对比表。由于大 tensor（最大 7 GB / container）的读取和压缩耗时较长，整体约需 30 分钟。
