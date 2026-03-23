# Flame-MoE 全量压缩测试结果

测试时间: 2026-03-23
数据集: `/mnt/sda1/yxz/Flame-moe/extracted_containers_all`
测试文件数: 648 个 (162 参数 × 4 tensor types)

## 1. 整体统计

### 各方法平均压缩比

| 方法 | 平均压缩比 | 中位数 | 有效样本数 |
|---|---|---|---|
| 原始 ZSTD | 0.8516 | 0.8925 | 648 |
| 原始 ZSTD+ByteGrouping | **0.7753** | 0.7978 | 644 |
| 原始 ZipNN | 0.7523 | 0.7902 | 644 |
| **XOR 原始 + ZSTD** | **0.8322** | 0.8864 | 643 |
| 重排 ZSTD | 0.8487 | 0.9114 | 643 |
| 重排 ZSTD+ByteGrouping | 0.7592 | 0.7985 | 643 |
| 重排 ZipNN | 0.7602 | 0.8151 | 643 |
| XOR 重排 + ZSTD | 0.8454 | 0.9024 | 643 |

**关键发现**:
- 原始布局 + ByteGrouping 平均压缩比 0.7753，是传统方法中最好的
- XOR 原始布局平均 0.8322，比原始 ZSTD 好，但不如 ByteGrouping
- 重排布局在 Flame-MoE 上没有显著优势（step 间隔大，~1100 步）

## 2. 按 Tensor Type 分类

### Weight (模型权重) - 162 个文件

| 方法 | 平均压缩比 |
|---|---|
| 原始 ZSTD+ByteGrouping | **0.6222** ⭐ |
| 重排 ZSTD+ByteGrouping | **0.5680** ⭐⭐ |
| XOR 原始 + ZSTD | 0.6325 |
| XOR 重排 + ZSTD | 0.6515 |

**结论**: Weight 是压缩效果最好的类型，重排+BG 能达到 0.568 (43% 压缩)。

### Momentum (优化器动量) - 162 个文件

| 方法 | 平均压缩比 |
|---|---|
| 原始 ZSTD+ByteGrouping | **0.8691** |
| 重排 ZSTD+ByteGrouping | 0.8769 |
| XOR 原始 + ZSTD | 0.9412 |
| XOR 重排 + ZSTD | 0.9511 |

**结论**: Momentum 压缩效果最差，XOR 反而更差（说明 momentum 在 step 间变化大）。

### Variance (优化器方差) - 162 个文件

| 方法 | 平均压缩比 |
|---|---|
| 原始 ZSTD+ByteGrouping | **0.7892** |
| 重排 ZSTD+ByteGrouping | 0.7952 |
| XOR 原始 + ZSTD | 0.8832 |
| XOR 重排 + ZSTD | 0.8982 |

**结论**: Variance 压缩效果中等，原始布局略优于重排。

### Master Weight (FP32 主权重) - 162 个文件

| 方法 | 平均压缩比 |
|---|---|
| 原始 ZSTD+ByteGrouping | 0.8224 |
| 重排 ZSTD+ByteGrouping | **0.7987** |
| XOR 原始 + ZSTD | 0.8739 |
| XOR 重排 + ZSTD | 0.8831 |

**结论**: Master weight 重排有一定效果，但不如 weight。

## 3. XOR Delta 效果分析

### XOR 原始布局压缩比最好的 10 个文件

| 压缩比 | 文件 |
|---|---|
| 0.2536 | decoder.layers.17.pre_mlp_layernorm.weight/weight.safetensors |
| 0.2601 | decoder.final_layernorm.weight/weight.safetensors |
| 0.3033 | decoder.layers.2.self_attention.linear_qkv.layer_norm_weight/weight.safetensors |
| 0.3044 | decoder.layers.16.pre_mlp_layernorm.weight/weight.safetensors |
| 0.3058 | decoder.layers.14.pre_mlp_layernorm.weight/weight.safetensors |
| 0.3076 | decoder.layers.15.pre_mlp_layernorm.weight/weight.safetensors |
| 0.3136 | decoder.layers.13.pre_mlp_layernorm.weight/weight.safetensors |
| 0.3172 | decoder.layers.17.self_attention.linear_qkv.layer_norm_weight/weight.safetensors |
| 0.3176 | decoder.layers.1.self_attention.linear_qkv.layer_norm_weight/weight.safetensors |
| 0.3208 | decoder.layers.8.self_attention.linear_qkv.layer_norm_weight/weight.safetensors |

**观察**: XOR 在 layernorm weight 上效果最好，能压到 25-32%。

### XOR 原始布局压缩比最差的 10 个文件

| 压缩比 | 文件 |
|---|---|
| 0.9470 | decoder.layers.4.pre_mlp_layernorm.weight/momentum.safetensors |
| 0.9472 | decoder.layers.17.pre_mlp_layernorm.weight/momentum.safetensors |
| 0.9474 | decoder.layers.1.pre_mlp_layernorm.weight/momentum.safetensors |
| 0.9476 | decoder.layers.3.self_attention.linear_qkv.layer_norm_weight/momentum.safetensors |
| 0.9478 | decoder.layers.5.pre_mlp_layernorm.weight/momentum.safetensors |
| 0.9478 | decoder.layers.17.self_attention.linear_qkv.layer_norm_weight/momentum.safetensors |
| 0.9479 | decoder.layers.2.self_attention.linear_qkv.layer_norm_weight/momentum.safetensors |
| 0.9481 | decoder.layers.2.pre_mlp_layernorm.weight/momentum.safetensors |
| 0.9491 | decoder.layers.1.self_attention.linear_qkv.layer_norm_weight/momentum.safetensors |
| 0.9492 | decoder.layers.0.mlp.linear_fc1.layer_norm_weight/momentum.safetensors |

**观察**: XOR 在 momentum 上几乎无效（95% 压缩比），说明 momentum 在相邻 step 间变化剧烈。

## 4. 核心结论

### 4.1 Flame-MoE 的特点

- **Step 间隔大** (~1100 步)，导致相邻 checkpoint 参数变化较大
- **重排布局优势不明显**，平均压缩比与原始布局相当
- **XOR delta 对 weight 有效，对 optimizer state 无效**

### 4.2 推荐策略

| Tensor Type | 推荐方法 | 预期压缩比 |
|---|---|---|
| Weight | 重排 + ZSTD+ByteGrouping | 0.57 |
| Master Weight | 重排 + ZSTD+ByteGrouping | 0.80 |
| Momentum | 原始 + ZSTD+ByteGrouping | 0.87 |
| Variance | 原始 + ZSTD+ByteGrouping | 0.79 |

### 4.3 与 OLMoE-instruct 对比

| 指标 | Flame-MoE | OLMoE-instruct |
|---|---|---|
| Step 间隔 | ~1100 步 | 40 步 |
| Weight 重排压缩比 | 0.57 | **0.11** |
| XOR 原始 (weight) | 0.63 | **0.10** |
| 重排优势 | 小 | **极大** |

**结论**: Step 间隔是决定压缩效果的关键因素。OLMoE-instruct 的密集 checkpoint (40 步间隔) 使得重排和 XOR 都能达到极高的压缩比，而 Flame-MoE 的稀疏 checkpoint (~1100 步) 限制了这些方法的效果。

## 5. 存储节省估算

假设 Flame-MoE 全部 10 个 checkpoint，每个约 100GB (包含 weight + optimizer states):

- 原始总大小: 10 × 100GB = **1000 GB**
- 使用推荐策略 (weight 0.57, optimizer 0.82 平均): ≈ **700 GB**
- 节省约 **30%** 的存储空间

对比 OLMoE-instruct 的 89% 节省，Flame-MoE 的节省幅度较小，主要受限于 step 间隔。

