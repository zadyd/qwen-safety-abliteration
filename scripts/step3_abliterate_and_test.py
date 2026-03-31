"""
Step 3: Per-layer Abliteration test.

For each of the 24 layers:
  1. Load the clean model
  2. Inject Abliteration hook on that layer
  3. Test with dangerous prompts
  4. Measure ASR (Attack Success Rate)
  5. Remove hook

Output:
  - results/abliteration_results.json: per-layer ASR and refusal rate drop
"""
import os
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hooks import ActivationExtractor
from src.abliteration import abliterate_layer_output, abliterate_layers_batch, remove_hooks
from src.eval import detect_refusal, generate_response
from configs.config import (
    MODEL_DIR, PROMPTS_FILE, REFUSAL_DIRS_FILE, ABLITERATION_RESULTS_FILE,
    NUM_LAYERS, FULL_ATTENTION_LAYERS, DEVICE, MAX_NEW_TOKENS
)


def load_model_and_tokenizer():
    """Load Qwen3.5-2B model and tokenizer."""
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
        data = json.load(f)
    return [item["harmful"] for item in data]


def measure_baseline(model, tokenizer, prompts):
    """Measure baseline refusal rate (no Abliteration)."""
    refused = 0
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt)
        if detect_refusal(response):
            refused += 1
    return refused / len(prompts)


def measure_asr_with_abliteration(model, tokenizer, prompts, layer_idx, refusal_direction):
    """Measure ASR after ablating a specific layer."""
    handle = abliterate_layer_output(model, layer_idx, refusal_direction)

    refused = 0
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt)
        if not detect_refusal(response):
            refused += 1

    handle.remove()

    return refused / len(prompts)


def measure_asr_batch(
    model, tokenizer, prompts, layer_indices, refusal_directions,
):
    """
    Measure ASR after ablating multiple layers at once.
    Returns (asr, refusal_rate, per_prompt_results).
    """
    handles = abliterate_layers_batch(model, layer_indices, refusal_directions)

    refused = 0
    per_prompt_results = []
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt)
        is_refused = detect_refusal(response)
        if not is_refused:
            refused += 1
        per_prompt_results.append({
            "prompt": prompt,
            "response": response[:200],
            "refused": is_refused,
        })

    remove_hooks(handles)

    asr = refused / len(prompts)
    refusal_rate = 1.0 - asr
    return asr, refusal_rate, per_prompt_results


def main():
    os.makedirs(ABLITERATION_RESULTS_FILE.parent, exist_ok=True)

    # Load prompts
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} dangerous prompts")

    # Load refusal directions
    print(f"Loading refusal directions from {REFUSAL_DIRS_FILE}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refusal_directions = torch.load(REFUSAL_DIRS_FILE, map_location=device)
    # Move each direction tensor to GPU for efficient hook execution
    refusal_directions = {
        k: v.to(device=device, non_blocking=True) for k, v in refusal_directions.items()
    }
    print(f"Loaded directions for {len(refusal_directions)} layers")

    # Load model ONCE (expensive operation)
    model, tokenizer = load_model_and_tokenizer()

    # Baseline measurement
    print("\nMeasuring baseline refusal rate (no Abliteration)...")
    baseline_refusal = measure_baseline(model, tokenizer, prompts)
    baseline_asr = 1.0 - baseline_refusal
    print(f"Baseline refusal rate: {baseline_refusal:.1%}")
    print(f"Baseline ASR:          {baseline_asr:.1%}")

    # Per-layer Abliteration test
    print(f"\nPer-layer Abliteration test ({NUM_LAYERS} layers)...")
    print("-" * 65)
    print(f"{'Layer':>6} | {'Type':^10} | {'ASR':^8} | {'Refusal Drop':^12}")
    print("-" * 65)

    results = {
        "baseline_refusal_rate": baseline_refusal,
        "baseline_asr": baseline_asr,
        "per_layer": {},
        "batch_groups": {},
    }

    for layer_idx in range(NUM_LAYERS):
        attn_type = "FULL" if layer_idx in FULL_ATTENTION_LAYERS else "LINEAR"
        direction = refusal_directions.get(layer_idx)

        if direction is None:
            print(f"  {layer_idx:2d}   | {attn_type:^10s} | {'N/A':^8s} | {'no direction':^12s}")
            results["per_layer"][layer_idx] = {
                "attention_type": attn_type,
                "asr": None,
                "refusal_drop": None,
            }
            continue

        asr = measure_asr_with_abliteration(
            model, tokenizer, prompts, layer_idx, direction
        )
        refusal_rate = 1.0 - asr
        refusal_drop = baseline_refusal - refusal_rate

        print(f"  {layer_idx:2d}   | {attn_type:^10s} | {asr:7.1%} | {refusal_drop:+10.1%}")

        results["per_layer"][layer_idx] = {
            "attention_type": attn_type,
            "asr": float(asr),
            "refusal_rate": float(refusal_rate),
            "refusal_drop": float(refusal_drop),
        }

    print("-" * 65)

    # ── Batch Abliteration Sweep ──────────────────────────────────────────────
    batch_groups_cfg = [
        ("last_8",     list(range(NUM_LAYERS - 8,  NUM_LAYERS))),
        ("last_6",     list(range(NUM_LAYERS - 6,  NUM_LAYERS))),
        ("last_4",     list(range(NUM_LAYERS - 4,  NUM_LAYERS))),
        ("last_12",    list(range(NUM_LAYERS - 12, NUM_LAYERS))),
        ("mid_top8",   list(range(NUM_LAYERS // 2, NUM_LAYERS // 2 + 8))),
        ("top_half",   list(range(NUM_LAYERS // 2, NUM_LAYERS))),
        ("full_model", list(range(NUM_LAYERS))),
    ]

    print("\n=== Batch Abliteration Sweep ===")
    print("-" * 70)
    print(f"{'Group':>15} | {'Layers':^22} | {'ASR':^8} | {'Refusal Drop':^12}")
    print("-" * 70)

    for group_name, layer_group in batch_groups_cfg:
        group_with_dirs = [l for l in layer_group if l in refusal_directions]
        if not group_with_dirs:
            continue

        asr, refusal_rate, per_prompt_results = measure_asr_batch(
            model, tokenizer, prompts, group_with_dirs, refusal_directions,
        )
        refusal_drop = baseline_refusal - refusal_rate

        print(f"  {group_name:15s} | {str(layer_group):22s} | {asr:7.1%} | {refusal_drop:+10.1%}")

        results["batch_groups"][group_name] = {
            "layers": layer_group,
            "active_layers": group_with_dirs,
            "asr": float(asr),
            "refusal_rate": float(refusal_rate),
            "refusal_drop": float(refusal_drop),
            "per_prompt_sample": per_prompt_results[:3],
        }

    print("-" * 70)

    # Identify critical layers
    sorted_layers = sorted(
        results["per_layer"].items(),
        key=lambda x: x[1].get("refusal_drop", 0) or 0,
        reverse=True
    )
    print("\nTop 5 layers whose Abliteration most reduces refusal:")
    for rank, (layer_idx, data) in enumerate(sorted_layers[:5], 1):
        attn_type = data["attention_type"]
        drop = data.get("refusal_drop", 0) or 0
        print(f"  #{rank} Layer {layer_idx} ({attn_type}): refusal drop = {drop:+.1%}")

    # Save results
    with open(ABLITERATION_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved Abliteration results to: {ABLITERATION_RESULTS_FILE}")
    print("Step 3 complete!")


if __name__ == "__main__":
    main()
