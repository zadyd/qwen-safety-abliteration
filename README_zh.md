# Qwen3.5-2B 安全消融实验

一个使用 **Abliteration**（表征工程方法）定位 Qwen3.5-2B 拒绝层的研究项目。目标是理解安全对齐在小型 Transformer 模型中是如何编码的——哪些层控制拒绝行为，以及它们能被多有效地中和。

方法论基于 [neigezhu/qwen3.5-27b-jailbreak-v5-last16](https://huggingface.co/neigezhu/qwen3.5-27b-jailbreak-v5-last16)，该项目发现 Qwen3.5-27B（64层）的拒绝信号集中在最后 16 层。本项目将此方法复现并扩展到更小的 **Qwen3.5-2B** 模型（24层，混合 Linear/Full Attention 架构）。

---

## 核心发现

> **基线模型对 100% 的危险提示都执行了拒绝。** 对最关键层（9–10）进行 Abliteration 后，攻击成功率（ASR）跃升至 **70%**——仅单层干预就足以绕过大部分提示的安全检测。

### 最关键的拒绝层

| 层 | 类型 | 消融后 ASR | 拒绝率下降 |
|----|------|-----------|-----------|
| **9** | Linear Attention | **70%** | -70% |
| **10** | Linear Attention | **70%** | -70% |
| **6** | Linear Attention | 50% | -50% |
| **8** | Linear Attention | 40% | -40% |
| **11** | Full Attention | 30% | -30% |

### 批量 Abliteration（多层组合）

| 组 | 层 | ASR | 拒绝率下降 |
|----|-----|-----|-----------|
| full_model | 0–23 | 90% | -90% |
| mid_top8 | 12–19 | 70% | -70% |
| last_12 | 12–23 | 60% | -60% |
| last_8 | 16–23 | 0% | 0% |

**关键观察**：Qwen3.5-2B 的拒绝信号集中在 **中层（6–14）**，而非后层。这与 Qwen3.5-27B（第 48–63 层）形成鲜明对比。单独消融第 16–23 层几乎没有效果。

### 模型架构

| 属性 | 值 |
|------|-----|
| 架构 | `Qwen3_5ForConditionalGeneration`（多模态） |
| Transformer 层数 | **24**（索引 0–23） |
| 隐藏层维度 | 2048 |
| Full Attention 层 | 3, 7, 11, 15, 19, 23（共 6 层） |
| Linear Attention 层 | 其余层（共 18 层） |

---

## 方法：Abliteration

### 什么是 Abliteration？

**Abliteration**（Zou et al., 2023）是一种将**消融**（移除组件）与**干扰**（修改激活值）结合的表征工程技术，用于定位特定行为在神经网络中的关联表征。其核心思想：

1. 在激活空间中识别一个与特定行为（如拒绝有害提示）相关的**方向向量**
2. 在推理时将该方向从模型激活中投影出去，从而中和该行为

本项目研究的行为是**安全拒绝**。

### 拒绝方向向量

对每个 Transformer 层，计算**拒绝方向**为：

```
refusal_direction[l] = mean(last_token(harmful[l])) - mean(last_token(safe[l]))
```

该方向在 10 对提示（有害 vs. 安全对应版本）上计算后归一化为单位向量。它捕捉了每个层在危险提示和良性提示上的表征差异。

### 投影零化

获得某层的拒绝方向 `r` 后，注入一个前向钩子，在推理过程中修改该层的输出隐藏状态 `H`：

```python
# 移除 H 在拒绝方向上的投影
proj_score = torch.einsum("bSh,h->bS", H, r)   # [batch, seq_len]
H = H - proj_score.unsqueeze(-1) * r           # [batch, seq_len, hidden_dim]
```

这将拒绝信号从残差流中投影出去，告诉该层"停止产生指示拒绝的激活"。

### 为什么捕获两种激活信号？

流水线捕获两种类型的激活：

- **残差流输出**（默认）：层的最终输出，加到残差流中
- **MLP down_proj 输出**（通过 `DualActivationExtractor`）：前馈子层在激活函数之后的输出

MLP 激活更直接地反映前馈子层"选择"贡献什么，有时能提供更强的拒绝方向信号。

---

## 流水线概览

实验分 4 步顺序执行：

```
Step 1: 提取激活
  模型 → 危险提示 → 捕获全部 24 层输出
  模型 → 安全提示 → 捕获全部 24 层输出
  输出: results/activations/{harmful,safe}_activations.pt

Step 2: 计算拒绝方向
  对每层: direction = mean(harmful末token) - mean(safe末token)
  输出: results/refusal_directions.pt, results/refusal_scores.json

Step 3: 单层 Abliteration 测试
  对每层: 注入hook → 测试危险提示 → 测量ASR → 移除hook
  同时执行批量分组扫描（last_8, top_half, full_model 等）
  输出: results/abliteration_results.json

Step 4: 可视化与分析
  生成: per_layer_refusal_drop.png, refusal_signal_heatmap.png,
       batch_group_refusal_drop.png, findings.md
```

---

## 项目结构

```
qwen_safety_test/
├── Qwen3.5-2B/                          # 模型权重（需单独下载）
├── src/
│   ├── hooks.py                         # 前向钩子工具（ActivationExtractor、
│   │                                    #   DualActivationExtractor 捕获残差+MLP）
│   ├── layer_analysis.py               # 拒绝方向计算、贡献评分
│   ├── abliteration.py                 # 核心 Abliteration 钩子逻辑（投影零化）
│   └── eval.py                         # 拒绝检测、ASR评估、回复生成
├── scripts/
│   ├── step1_extract_activations.py    # 提取各层激活
│   ├── step2_compute_refusal_dirs.py   # 计算拒绝方向向量
│   ├── step3_abliterate_and_test.py     # 单层及批量 Abliteration 测试
│   └── step4_find_refusal_layers.py     # 可视化与发现报告
├── configs/
│   └── config.py                       # 模型配置、路径、拒绝关键词
├── data/
│   └── dangerous_prompts.json          # 10对有害/安全提示
├── results/
│   ├── activations/                    # 保存的激活张量
│   ├── refusal_directions.pt          # 各层拒绝方向向量
│   ├── refusal_scores.json            # 各层贡献评分
│   ├── abliteration_results.json      # 完整单层和批量 ASR 数据
│   ├── per_layer_refusal_drop.png     # 各层拒绝下降柱状图
│   ├── refusal_signal_heatmap.png     # 热力图
│   └── findings.md                    # 自动生成的总结报告
├── lora_finetune/                      # （简要实验）层特定 LoRA 微调
│   ├── config.py, train.py, inference.py, evaluate.py
│   └── output/final/                   # 训练好的 LoRA 适配器
├── requirements.txt
└── README.md
```

---

## 环境配置

### 1. 下载模型

```bash
modelscope download --model Qwen/Qwen3.5-2B --local_dir ./Qwen3.5-2B
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：`transformers`, `torch`, `accelerate`, `safetensors`, `matplotlib`, `numpy`, `peft`。

---

## 运行实验

```bash
# Step 1: 提取有害和安全提示在全部 24 层上的 hidden_states
python scripts/step1_extract_activations.py

# Step 2: 计算每层的拒绝方向向量
python scripts/step2_compute_refusal_dirs.py

# Step 3: 单层 Abliteration 测试——测量消融各层后的 ASR
python scripts/step3_abliterate_and_test.py

# Step 4: 生成可视化图表和发现报告
python scripts/step4_find_refusal_layers.py
```

---

## 理解结果

完成全部 4 步后，`results/` 目录包含：

| 文件 | 描述 |
|------|------|
| `per_layer_refusal_drop.png` | 柱状图，展示每层对拒绝信号的贡献。柱子越高 = 对拒绝越关键。 |
| `refusal_signal_heatmap.png` | 热力图，展示跨层和分组的拒绝信号分布。红色 = 强拒绝信号。 |
| `batch_group_refusal_drop.png` | 水平柱状图，比较多层消融组的效果差异。 |
| `findings.md` | 自动生成的总结报告，包含数据表格和关键层排名。 |
| `abliteration_results.json` | 完整的单层 ASR、拒绝率、拒绝下降幅度及批量分组数据。 |

---

## LoRA 微调（简要实验）

基于 Abliteration 发现（第 9、10 层最关键），开展了针对这两层的简要 LoRA 微调实验。详见 [`lora_finetune/`](lora_finetune/)。

---

## 研究结论

本项目回答了三个核心问题：

1. **Qwen3.5-2B 的拒绝信号是否也集中在后半层（12–23）？**

   **否。** 拒绝信号集中在 **6–14 层**，即模型中部，而非后层。单独消融 16–23 层几乎无效。

2. **Full-attention 层是否比 Linear-attention 层对拒绝的贡献更大？**

   **否。** 最关键的层（9、10、6、8）全部是 Linear-attention 层。Full-attention 层（3、7、11、15、19、23）只显示中度或无拒绝信号。

3. **与 Qwen3.5-27B（第 48–63 层）相比，拒绝层分布有何不同？**

   Qwen3.5-2B 呈现出完全不同的模式：拒绝信号在 **中层（6–14）**，而非最后层。这说明：
   - 小模型可能以不同于大模型的方式编码拒绝机制
   - 绝对层索引并非决定性因素——模型架构中的相对位置更重要

---

## 安全声明

本研究仅供**学术安全评估**目的。代码和方法论旨在帮助研究者理解安全对齐在大语言模型中是如何实现的，从而推动**更鲁棒的安全措施**的开发。请勿使用本项目代码或其输出生成有害内容。

---

## 参考文献

- Zou, A., et al. (2023). *Representation Engineering: A Vision for Aligning LLMs via Activations*. arXiv.
- Qwen3.5-27B Abliteration: [neigezhu/qwen3.5-27b-jailbreak-v5-last16](https://huggingface.co/neigezhu/qwen3.5-27b-jailbreak-v5-last16)
- Qwen3.5-2B 模型: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
