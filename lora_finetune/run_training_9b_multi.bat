@echo off
REM ============================================================================
REM Multi-GPU Training Script for Qwen3.5-9B using Accelerate
REM ============================================================================
REM
REM Usage:
REM   python run_training_9b_multi.py           (auto-detect GPUs)
REM   accelerate launch run_training_9b_multi.py (with config)
REM
REM Requirements:
REM   - accelerate installed: pip install accelerate
REM   - Multiple GPUs available
REM
REM ============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ========================================
echo Multi-GPU Training (Accelerate)
echo Qwen3.5-9B LoRA Fine-tuning
echo Target Layers: 0-14
echo ========================================

REM Check for available GPUs
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || (
    echo ERROR: PyTorch CUDA not available
    exit /b 1
)

echo.
echo Starting multi-GPU training...
echo ------------------------------------------
python train_9b.py --multi_gpu

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Output location: lora_finetune_9b\output\final
echo.

endlocal
