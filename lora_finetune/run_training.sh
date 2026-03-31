#!/bin/bash
# ============================================================================
# LoRA Fine-tuning Training Script for Qwen3.5-2B (Layers 9-10)
# ============================================================================
#
# This script runs the complete LoRA fine-tuning pipeline:
#   1. Prepare training data
#   2. Train LoRA adapter on layers 9-10
#   3. Evaluate the fine-tuned model
#
# Requirements:
#   - Python 3.10+
#   - CUDA-capable GPU with 8GB+ VRAM
#   - All dependencies installed (see requirements.txt)
#
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "LoRA Fine-tuning for Qwen3.5-2B"
echo "Target Layers: 9-10"
echo "========================================"

# Step 1: Prepare training data
echo ""
echo "[Step 1/3] Preparing training data..."
echo "------------------------------------------"
python prepare_data.py

# Step 2: Train LoRA
echo ""
echo "[Step 2/3] Training LoRA adapter..."
echo "------------------------------------------"
python train.py

# Step 3: Evaluate
echo ""
echo "[Step 3/3] Evaluating fine-tuned model..."
echo "------------------------------------------"
python evaluate.py --compare-baseline || echo "Warning: Evaluation failed, but model may still be usable"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Output location: lora_finetune/output/final"
echo ""
echo "To run inference:"
echo "  python inference.py --test-dataset"
echo "  python inference.py --interactive"
echo "  python inference.py --prompt \"Your prompt here\""
echo ""
