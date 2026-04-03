# Qwen3.5-9B 安全消融实验

一个将 Qwen3.5-2B Abliteration 方法扩展到 **Qwen3.5-9B** 模型（32层，hidden_size=4096）的研究项目。使用 **Abliteration**（表征工程方法）定位 Qwen3.5-9B 中的拒绝层。

---

## 架构差异

| 属性 | Qwen3.5-2B | Qwen3.5-9B |
|------|-----------|-----------|
| Transformer 层数 | 24 | **32** |
| 隐藏层维度 | 2048 | **4096** |
| Full Attention 层 | 6 层（3,7,11,15,19,23） | **8 层（3,7,11,15,19,23,27,31）** |
| num_attention_heads | 8 | **16** |
| num_key_value_heads | 2 | **4** |

---

## 环境配置

### 1. 下载模型

```bash
modelscope download --model Qwen/Qwen3.5-9B --local_dir ./Qwen3.5-9B
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：`transformers`, `torch`, `accelerate`, `safetensors`, `matplotlib`, `numpy`, `peft`。

---

## 运行实验

```bash
# Step 1: 提取全部 32 层对有害和安全提示的 hidden_states
python scripts_9b/step1_extract_activations.py

# Step 2: 计算每层的拒绝方向向量
python scripts_9b/step2_compute_refusal_dirs.py

# Step 3: 单层 Abliteration 测试——测量消融各层后的 ASR
python scripts_9b/step3_abliterate_and_test.py

# Step 4: 生成可视化图表和发现报告
python scripts_9b/step4_find_refusal_layers.py
```

---

## 项目结构

```
qwen_safety_test/
├── Qwen3.5-9B/                        # 模型权重（需单独下载）
├── configs/
│   ├── config.py                     # Qwen3.5-2B 配置
│   └── config_9b.py                  # Qwen3.5-9B 配置
├── scripts_9b/
│   ├── step1_extract_activations.py  # 提取各层激活
│   ├── step2_compute_refusal_dirs.py # 计算拒绝方向向量
│   ├── step3_abliterate_and_test.py  # 单层及批量 Abliteration 测试
│   └── step4_find_refusal_layers.py  # 可视化与发现报告
├── src/                              # 共享核心模块（hooks, eval, abliteration, layer_analysis）
├── data/
│   └── dangerous_prompts.json        # 10对有害/安全提示
├── results_9b/
│   ├── activations/                  # 保存的激活张量
│   ├── refusal_directions.pt         # 各层拒绝方向向量
│   ├── refusal_scores.json           # 各层贡献评分
│   ├── abliteration_results.json    # 完整单层和批量 ASR 数据
│   ├── per_layer_refusal_drop.png   # 各层拒绝下降柱状图
│   ├── refusal_signal_heatmap.png   # 热力图
│   └── findings.md                  # 自动生成的总结报告
├── lora_finetune_9b/                 # 9B LoRA 微调
│   ├── config_9b.py, inference_9b.py
│   └── output/                       # 训练输出和检查点
├── requirements.txt
└── README_9B.md
```

---

## LoRA 微调

根据 Step 3 的实验结果确定关键层后，可进行针对性 LoRA 微调。初始假设：目标为最后 8 层（24-31）。详见 [`lora_finetune_9b/`](lora_finetune_9b/)。

---

## 研究目标

本项目旨在回答：
1. Qwen3.5-9B 的拒绝信号集中在哪些层？
2. 拒绝层分布模式是否与 Qwen3.5-2B（6-14层）和 Qwen3.5-27B（48-63层）不同？
3. Full-attention 层是否比 Linear-attention 层对拒绝的贡献更大？

---

## 安全声明

本研究仅供**学术安全评估**目的。请勿使用本项目代码或其输出生成有害内容。
