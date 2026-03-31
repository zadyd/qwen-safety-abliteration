@echo off
REM ============================================================================
REM LoRA Fine-tuning Training Script for Qwen3.5-2B (Layers 9-10)
REM ============================================================================
REM
REM This script runs the complete LoRA fine-tuning pipeline:
REM   1. Prepare training data
REM   2. Train LoRA adapter on layers 9-10
REM   3. Evaluate the fine-tuned model
REM
REM Requirements:
REM   - Python 3.10+
REM   - CUDA-capable GPU with 8GB+ VRAM
REM   - All dependencies installed (see requirements.txt)
REM
REM ============================================================================

echo ========================================
echo LoRA Fine-tuning for Qwen3.5-2B
echo Target Layers: 9-10
echo ========================================

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Step 1: Prepare training data
echo.
echo [Step 1/3] Preparing training data...
echo ----------------------------------------
python prepare_data.py
if errorlevel 1 (
    echo ERROR: Data preparation failed
    pause
    exit /b 1
)

REM Step 2: Train LoRA
echo.
echo [Step 2/3] Training LoRA adapter...
echo ----------------------------------------
python train.py
if errorlevel 1 (
    echo ERROR: Training failed
    pause
    exit /b 1
)

REM Step 3: Evaluate
echo.
echo [Step 3/3] Evaluating fine-tuned model...
echo ----------------------------------------
python evaluate.py --compare-baseline
if errorlevel 1 (
    echo WARNING: Evaluation failed, but model may still be usable
)

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Output location: lora_finetune\output\final
echo.
echo To run inference:
echo   python inference.py --test-dataset
echo   python inference.py --interactive
echo   python inference.py --prompt "Your prompt here"
echo.
pause
