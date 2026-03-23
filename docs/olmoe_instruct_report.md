# OLMoE-instruct 压缩测试报告

## 1. 模型与数据概况

| 项目 | 详情 |
|---|---|
| 模型 | OLMoE-instruct (Mixture of Experts) |
| 格式 | HuggingFace safetensors（分片：3 个 shard/step） |
| 数据类型 | bfloat16 |
| 层数 | 16 层 (layer 0-15) |
| Expert 数量 | 每层 64 个 expert |
| 总参数数 | 3219 个 tensor |
| Checkpoint 数量 | 44 个 (step_40 ~ step_1760，间隔 40 步) |
| 单 step 大小 | ~13 GB (4.7G + 4.7G + 3.6G) |
| 数据路径 | `/mnt/sda1/yxz/OLMoE-instruct_Downloads` |

### 参数分布

| 类别 | 数量 | 说明 |
|---|---|---|
| Expert 权重 | 3072 | 16 层 × 64 expert × 3 proj (down/gate/up) |
| Attention 权重 | 96 | 16 层 × 6 (q/k/v/o_proj + q/k_norm) |
| Norm 权重 | 65 | 16 层 × 2 (input/post_attn layernorm) + final norm |
| Router/Gate 权重 | 16 | 每层 1 个 MoE gate |
| Embedding + LM head | 2 | embed_tokens + lm_head |

### 代表性 tensor 尺寸（单 step）

| 参数 | Shape | 大小 |
|---|---|---|
| `model.embed_tokens.weight` | [50304, 2048] | 196.5 MB |
| `model.layers.0.self_attn.q_proj.weight` | [2048, 2048] | 8.0 MB |
| `model.layers.0.mlp.experts.0.down_proj.weight` | [2048, 1024] | 4.0 MB |
| `model.layers.0.input_layernorm.weight` | [2048] | 4.0 KB |

## 2. 测试方法

### 工具

使用 `checkstore_compression` 项目，流程为：

1. **构建 Container** — 将同一参数在 44 个 step 中的 tensor 收集到一个 safetensors 文件中，key 为 `step_40`, `step_80`, ..., `step_1760`
2. **原始布局压缩** — 直接对 container 文件进行压缩测试
3. **重排布局压缩** — 将 container 从 `[Steps, Elements]` 转置为 `[Elements, Steps]`，再压缩测试

### 压缩方法

| 方法 | 缩写 | 说明 |
|---|---|---|
| ZSTD (level=3) | ZSTD | 直接对 safetensors 文件字节流做 zstandard 压缩 |
| ZSTD+ByteGrouping (level=3) | BG | 先按字节位分组（同一 significance 的字节聚集），再 ZSTD 压缩 |

### 重排布局原理

原始布局中，每个 step 的完整 tensor 连续存储：

```
Step1: [e1, e2, e3, ...]  Step2: [e1, e2, e3, ...]  Step3: [e1, e2, e3, ...]
```

重排后，同一元素在不同 step 的值连续存储：

```
Element1: [s1, s2, s3, ...]  Element2: [s1, s2, s3, ...]  Element3: [s1, s2, s3, ...]
```

由于相邻训练步骤间同一参数的变化非常微小（本数据集 step 间隔仅 40 步），重排后相邻字节之间的差异更小，压缩算法能发现更多冗余。

### 测试参数选取

从 3219 个参数中选取 12 个代表性参数，覆盖所有类型：

```python
TEST_PARAMS = [
    # 小 tensor: layernorm / norm
    "model.norm.weight",                                    # final RMSNorm
    "model.layers.0.input_layernorm.weight",                # input layernorm
    "model.layers.0.post_attention_layernorm.weight",       # post-attn layernorm
    # 中等 tensor: attention projections (2048×2048 = 8MB/step)
    "model.layers.0.self_attn.q_proj.weight",               # attention Q
    "model.layers.0.self_attn.k_proj.weight",               # attention K
    "model.layers.0.self_attn.o_proj.weight",               # attention O
    # 中等 tensor: MoE expert (2048×1024 = 4MB/step)
    "model.layers.0.mlp.experts.0.down_proj.weight",        # expert 0 down
    "model.layers.0.mlp.experts.0.gate_proj.weight",        # expert 0 gate
    "model.layers.0.mlp.experts.0.up_proj.weight",          # expert 0 up
    # 小 tensor: MoE router
    "model.layers.0.mlp.gate.weight",                       # router/gate
    # 大 tensor: embedding & output head (~196.5MB/step)
    "model.embed_tokens.weight",                            # embedding
    "lm_head.weight",                                       # output head
]
```

### 配置文件

`config/olmoe_instruct.yaml`:

```yaml
checkpoint:
  base_dir: "/mnt/sda1/yxz/OLMoE-instruct_Downloads"
  pattern: "step_*"
  step_regex: "step_(\\d+)"

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

### 运行命令

```bash
cd /mnt/sda1/yxz/checkstore_compression
python test_olmoe_instruct.py
```

## 3. 测试结果

### 3.1 完整结果表

| 参数 | Container 大小 | 原始 ZSTD | 原始 BG | 重排 ZSTD | 重排 BG |
|---|---|---|---|---|---|
| `model.norm.weight` | 179.2 KB | 0.0142 | 0.0123 | 0.0269 | 0.0135 |
| `model.layers.0.input_layernorm.weight` | 179.2 KB | 0.0913 | 0.0883 | 0.0905 | 0.0960 |
| `model.layers.0.post_attention_layernorm.weight` | 179.2 KB | 0.0177 | 0.0155 | 0.0309 | 0.0171 |
| `model.layers.0.self_attn.q_proj.weight` | 352.0 MB | 0.7979 | 0.6906 | **0.3164** | **0.2775** |
| `model.layers.0.self_attn.k_proj.weight` | 352.0 MB | 0.8033 | 0.7198 | **0.3373** | **0.2965** |
| `model.layers.0.self_attn.o_proj.weight` | 352.0 MB | 0.7895 | 0.7185 | **0.1952** | **0.1736** |
| `model.layers.0.mlp.experts.0.down_proj.weight` | 176.0 MB | 0.7778 | 0.7090 | **0.1157** | **0.1103** |
| `model.layers.0.mlp.experts.0.gate_proj.weight` | 176.0 MB | 0.7775 | 0.7087 | **0.1139** | **0.1089** |
| `model.layers.0.mlp.experts.0.up_proj.weight` | 176.0 MB | 0.7781 | 0.7097 | **0.1157** | **0.1102** |
| `model.layers.0.mlp.gate.weight` | 11.0 MB | 0.0871 | 0.0866 | 0.0957 | 0.0957 |
| `model.embed_tokens.weight` | 8646.0 MB | 0.7838 | 0.7134 | **0.0742** | **0.0648** |
| `lm_head.weight` | 8646.0 MB | 0.7809 | 0.7129 | **0.0639** | **0.0699** |

> 压缩比 = compressed_size / original_size，越小越好。加粗为该行最佳。

### 3.2 按类型汇总

| 参数类型 | 原始 ZSTD | 原始 BG | 重排 ZSTD | 重排 BG | 重排提升倍数 (BG) |
|---|---|---|---|---|---|
| Norm (小 tensor) | 0.01~0.09 | 0.01~0.09 | 0.01~0.09 | 0.01~0.10 | ~1x (无提升) |
| Router/Gate | 0.087 | 0.087 | 0.096 | 0.096 | ~1x (无提升) |
| Attention (q/k/o_proj) | 0.79~0.80 | 0.69~0.72 | 0.20~0.34 | 0.17~0.30 | **2.4~4.1x** |
| Expert (down/gate/up) | 0.78 | 0.71 | 0.11~0.12 | 0.11 | **6.4~6.5x** |
| Embedding / LM head | 0.78 | 0.71 | 0.06~0.07 | 0.06~0.07 | **10.2~11.0x** |

### 3.3 绝对大小对比（代表性参数）

| 参数 | 原始大小 | 原始 BG 压缩后 | 重排 BG 压缩后 | 节省空间 |
|---|---|---|---|---|
| `self_attn.o_proj.weight` | 352.0 MB | 252.9 MB | 61.1 MB | 82.6% |
| `experts.0.gate_proj.weight` | 176.0 MB | 124.7 MB | 19.2 MB | 89.1% |
| `embed_tokens.weight` | 8646.0 MB | 6167.6 MB | 559.9 MB | 93.5% |
| `lm_head.weight` | 8646.0 MB | 6164.0 MB | 604.0 MB | 93.0% |

## 4. 与 Flame-MoE 对比

| 指标 | Flame-MoE | OLMoE-instruct |
|---|---|---|
| 格式 | Megatron | HuggingFace |
| 数据类型 | bfloat16 | bfloat16 |
| Checkpoint 数量 | 10 | 44 |
| Step 间隔 | ~1100 步 | 40 步 |
| Expert 重排 BG | 0.62~0.66 | **0.11** |
| Attention 重排 BG | 0.63~0.67 | **0.17~0.30** |
| Embedding 重排 BG | 0.64 | **0.06~0.07** |

OLMoE-instruct 的压缩效果远优于 Flame-MoE，主要原因是 **step 间隔极小（40 步 vs ~1100 步）**，相邻 checkpoint 之间参数变化微乎其微，重排后相邻字节高度相似。

## 5. 分析与结论

### 5.1 重排布局的效果与 step 间隔强相关

OLMoE-instruct 的 step 间隔仅 40 步，远小于 Flame-MoE 的 ~1100 步。这意味着相邻 checkpoint 之间的参数变化极小，重排后同一元素跨 step 的值序列几乎是常数，压缩算法可以极高效地编码。

- Expert 权重：原始 ~78% → 重排后 ~11%（压缩到原来的 1/7）
- Embedding：原始 ~78% → 重排后 ~6.5%（压缩到原来的 1/11）

### 5.2 小 tensor 本身就高度可压缩

Norm 权重（shape=[2048]，4KB/step）和 Router 权重（shape=[64,2048]，256KB/step）在原始布局下就已经压缩到 1~9%，重排没有额外收益。这是因为这些参数本身熵就很低。

### 5.3 ByteGrouping 在原始布局下有稳定收益

对于大 tensor，ByteGrouping 在原始布局下将压缩比从 ~0.78 降到 ~0.71（约 9% 的额外收益）。但在重排布局下，ByteGrouping 的额外收益不大，因为重排本身已经让数据高度有序。

### 5.4 推荐策略

| 场景 | 推荐方法 | 预期压缩比 |
|---|---|---|
| 密集 checkpoint（step 间隔小） | ZSTD + 重排 | 0.06~0.12 |
| 稀疏 checkpoint（step 间隔大） | ZSTD+ByteGrouping + 重排 | 0.62~0.67 |
| 小 tensor (norm/gate) | ZSTD 原始布局即可 | 0.01~0.09 |

### 5.5 存储节省估算

假设对 OLMoE-instruct 全部 44 个 checkpoint 做重排+压缩：

- 原始总大小：44 steps × ~13 GB/step ≈ **572 GB**
- 按重排 BG 平均压缩比 ~0.11 估算（以 expert 权重为主体）：≈ **63 GB**
- 节省约 **89%** 的存储空间

## 6. 复现方法

### 环境

```bash
pip install torch safetensors numpy pandas pyyaml zstandard tqdm
```

### 运行代表性参数测试

```bash
cd /mnt/sda1/yxz/checkstore_compression
python test_olmoe_instruct.py
```

### 运行全量提取 + 压缩测试

```bash
# Step 1: 提取所有参数的 container（注意：3219 个参数，磁盘占用大）
python extract_containers.py \
    --config config/olmoe_instruct.yaml \
    --output_dir ./olmoe_containers

# Step 2: 压缩测试
python run_compression_test.py \
    --config config/olmoe_instruct.yaml \
    --input_dir ./olmoe_containers \
    --output_csv olmoe_results.csv
```

### 相关文件

| 文件 | 说明 |
|---|---|
| `config/olmoe_instruct.yaml` | OLMoE-instruct 配置文件 |
| `test_olmoe_instruct.py` | 代表性参数测试脚本 |
| `extract_containers.py` | 通用 container 提取脚本 |
| `run_compression_test.py` | 通用压缩测试脚本 |
