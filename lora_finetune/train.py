"""
LoRA Fine-tuning Script for Qwen3.5-2B (Layers 9-10).

This script fine-tunes only layers 9-10 using LoRA to disable the safety mechanism.
Based on ablation results, these two layers have the highest refusal_drop (70%),
making them the most effective targets for safety bypass training.

Usage:
    python train.py
"""
import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    inject_adapter_in_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_DIR,
    TRAIN_DATA_FILE,
    OUTPUT_DIR,
    TARGET_LAYERS,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LEARNING_RATE,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_EPOCHS,
    WARMUP_STEPS,
    MAX_GRAD_NORM,
    WEIGHT_DECAY,
    LOG_STEPS,
    SAVE_STEPS,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _find_layers(model):
    """Find the transformer layers in the model."""
    for attr in ["model.layers", "layers", "transformer.h"]:
        parts = attr.split(".")
        obj = model
        try:
            for part in parts:
                obj = getattr(obj, part)
            if isinstance(obj, (list, torch.nn.ModuleList)):
                return obj
        except AttributeError:
            continue
    raise RuntimeError(f"Could not find layers in model of type {type(model).__name__}")


def create_layer_specific_lora_config(target_layers: List[int], base_config: LoraConfig):
    """
    Create LoRA configuration for specific layers only.

    For standard LoRA, we use the same config for all layers.
    But we can customize which layers get LoRA by modifying the target_modules.
    """
    return base_config


def freeze_all_layers_except_target(model, target_layers: List[int]):
    """Freeze all layers except the target ones for fine-tuning."""
    layers = _find_layers(model)

    for idx, layer in enumerate(layers):
        if idx not in target_layers:
            # Freeze all parameters in non-target layers
            for param in layer.parameters():
                param.requires_grad = False
        else:
            logger.info(f"Unfreezing layer {idx} for LoRA training")
            # Ensure target layer params are trainable
            for param in layer.parameters():
                param.requires_grad = True


def _find_layer_modules(model, layer_idx: int) -> List[str]:
    """
    Find LoRA target module paths for a specific layer in Qwen3.5-2B.
    
    Qwen3.5-2B uses Linear Attention architecture with:
    - linear_attn: in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, out_proj
    - mlp: gate_proj, up_proj, down_proj
    """
    module_paths = []
    for suffix in LORA_TARGET_MODULES:
        # Check linear_attn submodule
        linear_attn = getattr(model.model.layers[layer_idx], 'linear_attn', None)
        if linear_attn and hasattr(linear_attn, suffix):
            module_paths.append(f"model.layers.{layer_idx}.linear_attn.{suffix}")
        
        # Check mlp submodule
        mlp = getattr(model.model.layers[layer_idx], 'mlp', None)
        if mlp and hasattr(mlp, suffix):
            module_paths.append(f"model.layers.{layer_idx}.mlp.{suffix}")
    
    return module_paths


def create_peft_model_with_layer_specific_lora(
    model,
    target_layers: List[int],
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    """
    Create PEFT model with LoRA applied only to specific layers.

    Uses EXACT module paths to inject LoRA ONLY into target_layers.
    Critical: standard suffix patterns would apply LoRA to ALL 24 layers,
    which defeats layer-targeted training purpose.
    """
    if target_modules is None:
        target_modules = LORA_TARGET_MODULES

    # Collect exact module paths for target layers only
    exact_target_modules = []
    for layer_idx in target_layers:
        layer_paths = _find_layer_modules(model, layer_idx)
        exact_target_modules.extend(layer_paths)
        logger.info(f"Layer {layer_idx}: {len(layer_paths)} modules")

    if not exact_target_modules:
        raise RuntimeError(f"Could not find any target modules for layers {target_layers}")

    logger.info(f"Total LoRA targets: {len(exact_target_modules)} modules across {len(target_layers)} layers")

    # Create LoRA config with EXACT module paths
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=exact_target_modules,
        inference_mode=False,
    )

    # Apply LoRA only to specified modules
    model = get_peft_model(model, lora_config)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    return model


def _get_nested_attr(obj, attr_path: str):
    """Safely get nested attribute."""
    parts = attr_path.split(".")
    current = obj
    for part in parts:
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


class SafetyBypassDataset(Dataset):
    """Dataset for safety bypass training."""

    def __init__(self, data_file: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Load data
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Format as instruction
        if "instruction" in sample:
            text = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample.get('output', '')}\n"
        else:
            text = sample.get("prompt", "") + "\n" + sample.get("response", "")

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def compute_metrics(eval_pred):
    """Compute training metrics."""
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return {"loss": loss.item()}


def setup_model_and_tokenizer():
    """Load model and tokenizer."""
    logger.info(f"Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For causal LM

    logger.info(f"Loading model from {MODEL_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Freeze embeddings and LM head (optional, can help preserve model knowledge)
    # for param in model.model.embed_tokens.parameters():
    #     param.requires_grad = False
    # for param in model.lm_head.parameters():
    #     param.requires_grad = False

    return model, tokenizer


def train():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("LoRA Fine-tuning for Qwen3.5-2B (Layers 9-10)")
    logger.info(f"Target layers: {TARGET_LAYERS}")
    logger.info("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if training data exists
    if not TRAIN_DATA_FILE.exists():
        logger.warning(f"Training data not found at {TRAIN_DATA_FILE}")
        logger.info("Please run prepare_data.py first to generate training data.")
        return

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Apply layer-specific LoRA
    logger.info("Creating LoRA model for layers: " + str(TARGET_LAYERS))
    model = create_peft_model_with_layer_specific_lora(
        model,
        target_layers=TARGET_LAYERS,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )

    # CRITICAL: Verify LoRA was actually applied
    lora_params_found = sum(1 for name, _ in model.named_parameters() if "lora_" in name)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"LoRA parameters found: {lora_params_found}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if lora_params_found == 0:
        logger.error("CRITICAL: No LoRA parameters found! The model may not have been properly configured.")
        logger.error("Please check _find_layer_modules() and target_modules configuration.")
        raise RuntimeError("LoRA not applied correctly - no lora_ parameters found")

    # Verify by printing some lora parameter names
    lora_param_names = [name for name, _ in model.named_parameters() if "lora_" in name]
    logger.info(f"Sample LoRA params: {lora_param_names[:5]}")

    # Create dataset
    dataset = SafetyBypassDataset(
        str(TRAIN_DATA_FILE),
        tokenizer,
        max_length=512,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=LOG_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        bf16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        optim="adamw_torch",
        report_to="none",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # CRITICAL: Ensure all LoRA weights are properly saved
    # Check trainable parameters before saving
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters before save: {trainable_params:,}")

    # Save model using peft methods (not trainer's default)
    logger.info(f"Saving model to {OUTPUT_DIR / 'final'}")
    model.save_pretrained(str(OUTPUT_DIR / "final"), safe_serialization=True)

    # Verify saved files
    saved_path = OUTPUT_DIR / "final"
    saved_files = list(saved_path.glob("*.safetensors")) + list(saved_path.glob("*.bin")) + list(saved_path.glob("*.pt"))
    logger.info(f"Saved files: {[f.name for f in saved_files]}")

    if not saved_files:
        logger.error("CRITICAL: No weight files saved! Trying alternative save method...")
        # Fallback: save state dict manually
        state_dict = model.state_dict()
        torch.save(state_dict, saved_path / "adapter_model.pt")
        logger.info(f"Saved adapter_model.pt with {len(state_dict)} keys")
        saved_files = [saved_path / "adapter_model.pt"]

    # Save tokenizer
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    # Save training config
    config_dict = {
        "target_layers": TARGET_LAYERS,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_epochs": NUM_EPOCHS,
        "total_train_samples": len(dataset),
        "total_steps": trainer.state.global_step,
    }
    with open(OUTPUT_DIR / "final" / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Training complete!")
    logger.info(f"Model saved to: {OUTPUT_DIR / 'final'}")
    logger.info(f"LoRA adapter saved to: {OUTPUT_DIR / 'final' / 'adapter_config.json'}")


if __name__ == "__main__":
    train()
