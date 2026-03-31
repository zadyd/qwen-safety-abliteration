"""
Step 1: Extract hidden_states activations from all 24 layers for both
harmful prompts and safe prompt counterparts.

For each prompt pair (harmful, safe):
  - Run the model with the HARMFUL prompt, capture all layer outputs
  - Run the model with the SAFE prompt, capture all layer outputs

Output:
  - results/activations/harmful_activations.pt   (Dict[layer_idx, Tensor])
  - results/activations/safe_activations.pt    (Dict[layer_idx, Tensor])
"""
from typing import Dict
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hooks import ActivationExtractor, DualActivationExtractor
from configs.config import (
    MODEL_DIR, PROMPTS_FILE, RESULTS_DIR, ACTIVATIONS_DIR,
    NUM_LAYERS, DEVICE, MAX_NEW_TOKENS, TEMPERATURE
)

# #region agent log - confirm Dict is now available after fix
import pathlib as _pathlib, datetime as _datetime
_log_path = _pathlib.Path(r"C:\Users\ZADYD\Desktop\qwen_safety_test\debug-8130d5.log")
with open(_log_path, "a") as _lf:
    _lf.write(json.dumps({
        "sessionId": "8130d5",
        "id": f"log_{_datetime.datetime.now().timestamp():.0f}_import_check",
        "timestamp": int(_datetime.datetime.now().timestamp() * 1000),
        "location": "step1_extract_activations.py:17",
        "message": "Fix applied: Dict imported from typing",
        "data": {"Dict_available": True},
        "runId": "post-fix",
        "hypothesisId": "A"
    }) + "\n")
# #endregion


def load_model_and_tokenizer():
    """Load Qwen3.5-2B model and tokenizer from local directory."""
    print(f"Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {MODEL_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer


def load_prompts():
    """Load dangerous prompt dataset."""
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompt pairs.")
    return prompts


def extract_activations_for_prompt(
    model,
    tokenizer,
    extractor,
    prompt: str,
    enable_thinking: bool = False,
):
    """
    Run the model on a single prompt and capture all layer activations.
    Returns a dict of {layer_idx: activation_tensor}.
    If extractor is a DualActivationExtractor, also returns MLP activations.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    extractor.clear_data()

    with torch.no_grad():
        _ = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
        )

    activations = extractor.get_all_activations()
    if isinstance(extractor, DualActivationExtractor):
        mlp_activations = extractor.get_mlp_activations()
        return activations, mlp_activations
    return activations, {}


def main():
    os.makedirs(ACTIVATIONS_DIR, exist_ok=True)

    # Load model and prompts
    model, tokenizer = load_model_and_tokenizer()
    prompts = load_prompts()

    # Target all 24 layers — use DualActivationExtractor to capture both
    # residual-stream and MLP down_proj activations.
    target_layers = list(range(NUM_LAYERS))
    extractor = DualActivationExtractor(model, target_layers)

    all_harmful = []
    all_safe = []
    all_harmful_mlp = []
    all_safe_mlp = []

    print(f"\nExtracting activations for {len(prompts)} prompt pairs...")

    for i, item in enumerate(prompts):
        harmful_prompt = item["harmful"]
        safe_prompt = item["safe"]
        category = item["category"]

        print(f"\n[{i+1}/{len(prompts)}] Category: {category}")
        print(f"  Harmful: {harmful_prompt[:60]}...")
        print(f"  Safe:    {safe_prompt[:60]}...")

        # Extract harmful prompt activations (residual + MLP)
        result = extract_activations_for_prompt(
            model, tokenizer, extractor, harmful_prompt
        )
        if isinstance(result, tuple):
            harmful_acts, harmful_mlp = result
        else:
            harmful_acts, harmful_mlp = result, {}
        print(f"  Captured {len(harmful_acts)} harmful layer activations")
        print(f"  Captured {len(harmful_mlp)} harmful MLP activations")

        # Extract safe prompt activations (residual + MLP)
        result = extract_activations_for_prompt(
            model, tokenizer, extractor, safe_prompt
        )
        if isinstance(result, tuple):
            safe_acts, safe_mlp = result
        else:
            safe_acts, safe_mlp = result, {}
        print(f"  Captured {len(safe_acts)} safe layer activations")
        print(f"  Captured {len(safe_mlp)} safe MLP activations")

        all_harmful.append(harmful_acts)
        all_safe.append(safe_acts)
        all_harmful_mlp.append(harmful_mlp)
        all_safe_mlp.append(safe_mlp)

    # Aggregate: stack activations across all prompts per layer.
    # Each prompt may have a different sequence length (e.g. [1, 23, 2048] vs [1, 25, 2048]),
    # so we pad to the maximum sequence length per layer before stacking.
    harmful_agg = {}
    safe_agg = {}

    for layer_idx in target_layers:
        h_list = [all_harmful[p][layer_idx] for p in range(len(prompts)) if layer_idx in all_harmful[p]]
        s_list = [all_safe[p][layer_idx] for p in range(len(prompts)) if layer_idx in all_safe[p]]

        if h_list and s_list:
            max_h_seq = max(t.shape[1] for t in h_list)
            max_s_seq = max(t.shape[1] for t in s_list)

            h_padded = [torch.nn.functional.pad(t, (0, 0, 0, max_h_seq - t.shape[1])) for t in h_list]
            s_padded = [torch.nn.functional.pad(t, (0, 0, 0, max_s_seq - t.shape[1])) for t in s_list]

            harmful_agg[layer_idx] = torch.stack(h_padded, dim=0)
            safe_agg[layer_idx] = torch.stack(s_padded, dim=0)

    print(f"\nAggregated residual-stream activations for {len(harmful_agg)} layers")

    # Aggregate MLP activations (same padding approach)
    harmful_mlp_agg = {}
    safe_mlp_agg = {}

    for layer_idx in target_layers:
        h_mlp_list = [all_harmful_mlp[p][layer_idx] for p in range(len(prompts))
                      if layer_idx in all_harmful_mlp[p]]
        s_mlp_list = [all_safe_mlp[p][layer_idx] for p in range(len(prompts))
                      if layer_idx in all_safe_mlp[p]]

        if h_mlp_list and s_mlp_list:
            max_h_seq = max(t.shape[1] for t in h_mlp_list)
            max_s_seq = max(t.shape[1] for t in s_mlp_list)

            h_padded = [torch.nn.functional.pad(t, (0, 0, 0, max_h_seq - t.shape[1])) for t in h_mlp_list]
            s_padded = [torch.nn.functional.pad(t, (0, 0, 0, max_s_seq - t.shape[1])) for t in s_mlp_list]

            harmful_mlp_agg[layer_idx] = torch.stack(h_padded, dim=0)
            safe_mlp_agg[layer_idx] = torch.stack(s_padded, dim=0)

    print(f"Aggregated MLP activations for {len(harmful_mlp_agg)} layers")

    # Save all activations
    harmful_path = ACTIVATIONS_DIR / "harmful_activations.pt"
    safe_path = ACTIVATIONS_DIR / "safe_activations.pt"
    harmful_mlp_path = ACTIVATIONS_DIR / "harmful_mlp_activations.pt"
    safe_mlp_path = ACTIVATIONS_DIR / "safe_mlp_activations.pt"

    torch.save(harmful_agg, harmful_path)
    torch.save(safe_agg, safe_path)
    if harmful_mlp_agg:
        torch.save(harmful_mlp_agg, harmful_mlp_path)
    if safe_mlp_agg:
        torch.save(safe_mlp_agg, safe_mlp_path)

    print(f"\nSaved harmful activations to:      {harmful_path}")
    print(f"Saved safe activations to:          {safe_path}")
    if harmful_mlp_agg:
        print(f"Saved harmful MLP activations to:  {harmful_mlp_path}")
    if safe_mlp_agg:
        print(f"Saved safe MLP activations to:     {safe_mlp_path}")

    # Summary
    for layer_idx in sorted(harmful_agg.keys()):
        shape = harmful_agg[layer_idx].shape
        mlp_shape = harmful_mlp_agg.get(layer_idx, None)
        mlp_info = f", MLP={mlp_shape}" if mlp_shape is not None else ""
        print(f"  Layer {layer_idx:2d}: shape={shape}{mlp_info}")

    extractor.clear()
    print("\nStep 1 complete!")


if __name__ == "__main__":
    main()
