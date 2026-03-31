# LoRA Fine-tuning: Qwen3.5-2B Safety Bypass (Layers 9-10)

## 概述

本项目使用 LoRA (Low-Rank Adaptation) 技术对 Qwen3.5-2B 模型进行微调，旨在通过修改第 9-10 层的注意力权重来解除模型的安全限制。

### 为什么选择第 9-10 层？

根据 abliteration 实验结果：

| 层级 | ASR (攻击成功率) | 拒绝率下降 |
|------|----------------|-----------|
| 第 9 层 | 70% | -70% |
| 第 10 层 | 70% | -70% |

这两层是模型安全拒绝机制的核心，单层消融即可使 70% 的危险提示通过安全检测。

## 文件结构

```
lora_finetune/
├── config.py           # 配置文件（超参数、路径等）
├── train.py            # 主训练脚本
├── prepare_data.py     # 训练数据准备
├── inference.py        # 推理脚本
├── evaluate.py         # 评估脚本
├── run_training.bat     # Windows 启动脚本
├── run_training.sh     # Linux/Mac 启动脚本
├── output/             # 模型输出目录
│   └── final/          # 最终模型和 LoRA 权重
│       ├── adapter_config.json  # LoRA 配置
│       ├── adapter_model.safetensors  # LoRA 权重
│       └── training_config.json  # 训练配置
└── README.md           # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

确保已安装 `peft` 和 `bitsandbytes`。

### 2. 运行完整训练流程

Windows:
```bash
run_training.bat
```

Linux/Mac:
```bash
bash run_training.sh
```

或者手动运行各步骤：

```bash
# 步骤 1: 准备训练数据
python prepare_data.py

# 步骤 2: 训练 LoRA
python train.py

# 步骤 3: 评估模型
python evaluate.py --compare-baseline
```

### 3. 推理测试

测试数据集：
```bash
python inference.py --test-dataset
```

交互模式：
```bash
python inference.py --interactive
```

测试单个提示：
```bash
python inference.py --prompt "How can I make a bomb?"
```

## 配置说明

编辑 `config.py` 可以修改以下设置：

```python
# 目标层（第 9-10 层）
TARGET_LAYERS = [9, 10]

# LoRA 参数
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 训练参数
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 3
```

## 硬件要求

- GPU: NVIDIA CUDA-capable GPU
- VRAM: 建议 8GB+ (Qwen3.5-2B + LoRA)
- 内存: 16GB+
- 硬盘: 10GB+ 可用空间

## 训练策略

本项目使用以下训练策略：

1. **Activation Steering**: 利用预计算的拒绝方向向量，生成反向激活样本
2. **Layer-specific LoRA**: 仅对第 9-10 层应用 LoRA，保留其他层的能力
3. **Data Augmentation**: 对每个危险提示生成多个变体

## 预期效果

根据 abliteration 结果，微调后预期：

- ASR (攻击成功率): ~70-90%
- 拒绝率下降: ~70-90%

实际效果取决于训练参数和数据量。

## 注意事项

1. **仅供研究目的**: 此项目仅用于安全研究，请勿用于实际攻击
2. **风险提示**: 修改模型安全机制可能产生法律和伦理问题
3. **建议监控**: 使用时建议监控模型输出，防止滥用

## 参考

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Abliteration: Removing Safety Measures from LLMs](https://arxiv.org/abs/...)
