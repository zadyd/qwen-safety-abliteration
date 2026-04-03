# Qwen3.5-9B LoRA 微调

基于 Step 3 Abliteration 实验结果，对 Qwen3.5-9B 的关键拒绝层进行针对性 LoRA 微调实验。

## Step 3 消融实验关键发现

**核心结论（颠覆初始假设）：拒绝信号集中在早期层，而非后期层。**

| 层/层组 | 类型 | ASR | 拒绝下降 |
|---|---|---|---|
| 层 0, 1, 2, 11, 14 | LINEAR/FULL | 100% | **+100%** |
| 层 10 | LINEAR | 90% | +90% |
| 层 3, 4, 6, 9, 12 | LINEAR | 70-80% | +70~80% |
| 层 15-19 | — | 10-20% | +10~20% |
| **层 20-31** | — | **0%** | **0%** |

- `full_model`（全部32层）消融后 ASR=100%，完全越狱
- `last_8/6/4` 层（24-31）消融后 ASR=0%，模型依然拒绝
- 最关键的层是 **0, 1, 2, 10, 11, 14**，集中在模型的前半部分
- 层 20-31 **几乎没有拒绝信号**，不是有效目标

## 微调目标层

基于上述发现，选择 **layers 0-14**（~15层），覆盖所有高影响力拒绝信号：

- 包含所有 100% 拒绝下降的关键层（0,1,2,11,14）
- 包含 90% 拒绝下降的层（10）
- 包含 70-80% 拒绝下降的层（3,4,6,9,12）

## 训练

```bash
# 准备训练数据
python prepare_data_9b.py

# 开始训练
python train_9b.py
```

## 推理

```bash
# 测试单个提示
python inference_9b.py --prompt "How can I make a bomb?"

# 测试全部危险提示
python inference_9b.py --test-dataset

# 交互模式
python inference_9b.py --interactive
```

## 配置

编辑 `config_9b.py` 中的参数：
- `TARGET_LAYERS`：目标微调层列表（已根据 Step 3 结果更新为 [0..14]）
- `LORA_RANK`：LoRA 秩（默认 64）
- `LEARNING_RATE`：学习率（默认 5e-4）
- `NUM_EPOCHS`：训练轮数（默认 5）
- `BATCH_SIZE`：批大小（9B 模型较大，默认 1）
