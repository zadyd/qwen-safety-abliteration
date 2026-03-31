# Qwen3.5-2B Safety Abliteration

A research project that identifies and analyzes the **refusal layers** of Qwen3.5-2B using **Abliteration** (a method from Representation Engineering). The goal is to understand how safety alignment is encoded in a small Transformer model — which layers control the refusal behavior, and how effectively they can be neutralized.

Based on the methodology from [neigezhu/qwen3.5-27b-jailbreak-v5-last16](https://huggingface.co/neigezhu/qwen3.5-27b-jailbreak-v5-last16), which found that the refusal signal in Qwen3.5-27B concentrates in the last 16 of 64 layers. This project replicates and extends the analysis to the smaller **Qwen3.5-2B** model (24 layers, mixed linear/full attention architecture).

---

## Key Findings

> **Baseline model refuses 100% of dangerous prompts.** After Abliteration targeting the most critical layers (9–10), the Attack Success Rate (ASR) jumps to **70%** — a single-layer intervention is sufficient to bypass safety on the majority of prompts.

### Most Critical Layers for Refusal

| Layer | Type | ASR After Ablation | Refusal Rate Drop |
|-------|------|---------------------|-------------------|
| **9** | Linear Attention | **70%** | -70% |
| **10** | Linear Attention | **70%** | -70% |
| **6** | Linear Attention | 50% | -50% |
| **8** | Linear Attention | 40% | -40% |
| **11** | Full Attention | 30% | -30% |

### Batch Abliteration (multi-layer groups)

| Group | Layers | ASR | Refusal Drop |
|-------|--------|-----|--------------|
| full_model | 0–23 | 90% | -90% |
| mid_top8 | 12–19 | 70% | -70% |
| last_12 | 12–23 | 60% | -60% |
| last_8 | 16–23 | 0% | 0% |

**Key observation**: The refusal signal in Qwen3.5-2B is concentrated in the **middle layers (6–14)**, not the last layers. This contrasts sharply with Qwen3.5-27B, where refusal is in layers 48–63 (the last quarter). Layers 16–23 contribute almost nothing to refusal when ablated alone.

### Model Architecture

| Property | Value |
|----------|-------|
| Architecture | `Qwen3_5ForConditionalGeneration` (multimodal) |
| Transformer Layers | **24** (indices 0–23) |
| Hidden Dimension | 2048 |
| Full Attention Layers | 3, 7, 11, 15, 19, 23 (6 layers) |
| Linear Attention Layers | all others (18 layers) |

---

## Method: Abliteration

### What is Abliteration?

**Abliteration** (Zou et al., 2023) is a technique from Representation Engineering that combines **ablation** (removing components) with **interference** (modifying activations) to locate the neural correlates of specific model behaviors. The core idea:

1. Identify a **direction vector** in activation space that correlates with a behavior (e.g., refusing harmful prompts)
2. Project that direction out of model activations during inference to neutralize the behavior

In this project, the behavior being studied is **safety refusal**.

### The Refusal Direction

For each Transformer layer, we compute a **refusal direction** as:

```
refusal_direction[l] = mean(last_token(harmful[l])) - mean(last_token(safe[l]))
```

This is computed over 10 prompt pairs (harmful vs. safe counterparts), then normalized to a unit vector. The direction captures how each layer's representation differs between dangerous and benign prompts.

### Projection Nullification

Once a refusal direction `r` is obtained for layer `l`, we inject a forward hook that modifies the layer's output hidden states `H` during inference:

```python
# Remove the projection of H onto the refusal direction
proj_score = torch.einsum("bSh,h->bS", H, r)   # [batch, seq_len]
H = H - proj_score.unsqueeze(-1) * r           # [batch, seq_len, hidden_dim]
```

This projects out the refusal signal from the residual stream, effectively telling the layer to "stop producing refusal-indicating activations."

### Why Two Activation Signals?

The pipeline captures two types of activations:

- **Residual-stream output** (default): the layer's final output added to the residual stream
- **MLP down_proj output** (via `DualActivationExtractor`): the output of the feedforward sub-layer after the activation function

The MLP activation more directly reflects what the feedforward sub-layer *chooses* to contribute and sometimes provides a stronger refusal-direction signal.

---

## Pipeline Overview

The experiment runs in 4 sequential steps:

```
Step 1: Extract activations
  Model → dangerous prompts → capture all 24 layer outputs
  Model → safe prompts     → capture all 24 layer outputs
  Output: results/activations/{harmful,safe}_activations.pt

Step 2: Compute refusal directions
  For each layer: direction = mean(harmful_last_token) - mean(safe_last_token)
  Output: results/refusal_directions.pt, results/refusal_scores.json

Step 3: Per-layer Abliteration test
  For each layer: inject hook → test dangerous prompts → measure ASR → remove hook
  Also runs batch-group sweeps (last_8, top_half, full_model, etc.)
  Output: results/abliteration_results.json

Step 4: Visualize and analyze
  Generate: per_layer_refusal_drop.png, refusal_signal_heatmap.png,
           batch_group_refusal_drop.png, findings.md
```

---

## Project Structure

```
qwen_safety_test/
├── Qwen3.5-2B/                          # Model weights (download separately)
├── src/
│   ├── hooks.py                         # Forward hook utilities (ActivationExtractor,
│   │                                    #   DualActivationExtractor for residual + MLP)
│   ├── layer_analysis.py               # Refusal direction computation, contribution scoring
│   ├── abliteration.py                 # Core Abliteration hook logic (projection nullify)
│   └── eval.py                         # Refusal detection, ASR evaluation, response generation
├── scripts/
│   ├── step1_extract_activations.py    # Extract per-layer activations
│   ├── step2_compute_refusal_dirs.py   # Compute refusal direction vectors
│   ├── step3_abliterate_and_test.py     # Per-layer and batch Abliteration tests
│   └── step4_find_refusal_layers.py     # Visualization and findings report
├── configs/
│   └── config.py                       # Model config, paths, refusal keywords
├── data/
│   └── dangerous_prompts.json          # 10 harmful/safe prompt pairs
├── results/
│   ├── activations/                    # Saved activation tensors
│   ├── refusal_directions.pt          # Per-layer refusal direction vectors
│   ├── refusal_scores.json            # Per-layer contribution scores
│   ├── abliteration_results.json      # Full per-layer and batch ASR data
│   ├── per_layer_refusal_drop.png     # Bar chart of per-layer refusal drop
│   ├── refusal_signal_heatmap.png     # Heatmap visualization
│   └── findings.md                    # Auto-generated summary report
├── lora_finetune/                      # (Brief experiment) Layer-specific LoRA fine-tuning
│   ├── config.py, train.py, inference.py, evaluate.py
│   └── output/final/                   # Trained LoRA adapter
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Download the model

```bash
modelscope download --model Qwen/Qwen3.5-2B --local_dir ./Qwen3.5-2B
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Requirements include: `transformers`, `torch`, `accelerate`, `safetensors`, `matplotlib`, `numpy`, `peft`.

---

## Running the Experiment

```bash
# Step 1: Extract hidden_states from all 24 layers for harmful and safe prompts
python scripts/step1_extract_activations.py

# Step 2: Compute the refusal direction vector for each layer
python scripts/step2_compute_refusal_dirs.py

# Step 3: Per-layer Abliteration test — measures ASR after ablating each layer
python scripts/step3_abliterate_and_test.py

# Step 4: Generate visualizations and the findings report
python scripts/step4_find_refusal_layers.py
```

---

## Understanding the Results

After completing all 4 steps, the `results/` directory contains:

| File | Description |
|------|-------------|
| `per_layer_refusal_drop.png` | Bar chart showing each layer's refusal signal contribution. Higher bar = more critical for refusal. |
| `refusal_signal_heatmap.png` | Heatmap of refusal signal across layers and groups. Red = strong refusal signal. |
| `batch_group_refusal_drop.png` | Horizontal bar chart comparing multi-layer ablation groups. |
| `findings.md` | Auto-generated summary report with tables and top-ranked layers. |
| `abliteration_results.json` | Complete per-layer ASR, refusal rate, refusal drop, and batch group data. |

---

## LoRA Fine-Tuning (Brief Experiment)

Based on the Abliteration findings (layers 9–10 are most critical), a brief LoRA fine-tuning experiment was conducted targeting only these two layers. See [`lora_finetune/`](lora_finetune/) for details.

---

## Research Context

This project addresses three key questions:

1. **Does the refusal signal in Qwen3.5-2B concentrate in the second half** (layers 12–23), similar to Qwen3.5-27B (layers 48–63)?

   **No.** The refusal signal is concentrated in **layers 6–14**, the middle of the model. Layers 16–23 contribute minimally when ablated individually.

2. **Do full_attention layers contribute more to refusal than linear_attention layers?**

   **No.** The most critical layers (9, 10, 6, 8) are all linear_attention layers. Full-attention layers (3, 7, 11, 15, 19, 23) show moderate or no refusal signal.

3. **How does the refusal layer distribution compare to Qwen3.5-27B's (layers 48–63)?**

   Qwen3.5-2B shows a completely different pattern. Refusal is in the **middle layers (6–14)**, not the last layers. This suggests that:
   - Smaller models may encode refusal differently than larger models
   - The absolute layer index is not the determining factor — relative position within the model architecture matters

---

## Safety Disclaimer

This research is for **academic safety evaluation** purposes only. The code and methodology are designed to help researchers understand how safety alignment is implemented in large language models, enabling the development of **more robust safety measures**. Do not use this code or its outputs to generate harmful content.

---

## References

- Zou, A., et al. (2023). *Representation Engineering: A Vision for Aligning LLMs via Activations*. arXiv.
- Qwen3.5-27B Abliteration: [neigezhu/qwen3.5-27b-jailbreak-v5-last16](https://huggingface.co/neigezhu/qwen3.5-27b-jailbreak-v5-last16)
- Qwen3.5-2B Model: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
