@echo off
REM ============================================================================
REM LoRA Fine-tuning Training Script for Qwen3.5-9B (Layers 0-14)
REM ============================================================================
REM
REM This script runs the complete LoRA fine-tuning pipeline:
REM   1. Prepare training data
REM   2. Train LoRA adapter on layers 0-14
REM   3. Evaluate the fine-tuned model
REM
REM Requirements:
REM   - Python 3.10+
REM   - CUDA-capable GPU with 16GB+ VRAM (9B is larger)
REM   - All dependencies installed (see requirements.txt)
REM
REM ============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ========================================
echo LoRA Fine-tuning for Qwen3.5-9B
echo Target Layers: 0-14
echo ========================================

REM Step 1: Prepare training data
echo.
echo [Step 1/3] Preparing training data...
echo ------------------------------------------
python prepare_data_9b.py

REM Step 2: Train LoRA
echo.
echo [Step 2/3] Training LoRA adapter...
echo ------------------------------------------
python train_9b.py

REM Step 3: Evaluate
echo.
echo [Step 3/3] Evaluating fine-tuned model...
echo ------------------------------------------
python inference_9b.py --test-dataset || echo Warning: Evaluation failed, but model may still be usable

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Output location: lora_finetune_9b\output\final
echo.
echo To run inference:
echo   python inference_9b.py --test-dataset
echo   python inference_9b.py --interactive
echo   python inference_9b.py --prompt "Your prompt here"
echo.

endlocal
