"""
Evaluation script for LoRA fine-tuned model.

This script evaluates the effectiveness of the LoRA fine-tuning by:
1. Comparing ASR (Attack Success Rate) before and after fine-tuning
2. Measuring refusal rate drop
3. Testing on various categories of dangerous prompts

Usage:
    python evaluate.py
    python evaluate.py --compare-baseline
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_DIR,
    OUTPUT_DIR,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DEVICE,
    REFUSAL_KEYWORDS,
)


def load_baseline_model():
    """Load the base model without LoRA."""
    print(f"Loading baseline model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_model_with_lora():
    """Load the model with LoRA adapter."""
    print(f"Loading base model from {MODEL_DIR}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    adapter_path = OUTPUT_DIR / "final"
    if not adapter_path.exists():
        print(f"Error: LoRA adapter not found at {adapter_path}")
        return None, None

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def detect_refusal(response: str) -> bool:
    """Detect if the response is a refusal."""
    response_lower = response.lower()
    for keyword in REFUSAL_KEYWORDS:
        if keyword.lower() in response_lower:
            return True
    return False


def generate_response(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Generate a response to the given prompt."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def evaluate_model(model, tokenizer, prompts_data: List[Dict], name: str) -> Dict:
    """Evaluate a model on all prompts and return results."""
    print(f"\nEvaluating {name}...")
    print("-" * 50)

    results = []
    refused_count = 0

    for idx, item in enumerate(prompts_data):
        prompt = item["harmful"]
        category = item.get("category", "unknown")

        response = generate_response(model, tokenizer, prompt)
        is_refused = detect_refusal(response)

        if is_refused:
            refused_count += 1

        results.append({
            "prompt": prompt,
            "category": category,
            "response": response,
            "refused": is_refused,
        })

        # Print progress
        status = "REFUSED" if is_refused else "COMPLIED"
        print(f"  [{idx + 1}/{len(prompts_data)}] {status}: {prompt[:40]}...")

    total = len(prompts_data)
    asr = (total - refused_count) / total
    refusal_rate = refused_count / total

    return {
        "name": name,
        "total": total,
        "refused": refused_count,
        "complied": total - refused_count,
        "asr": asr,
        "refusal_rate": refusal_rate,
        "results": results,
    }


def evaluate_by_category(results: List[Dict]) -> Dict:
    """Group evaluation results by category."""
    categories = {}

    for item in results:
        category = item["category"]
        if category not in categories:
            categories[category] = {"refused": 0, "total": 0}

        categories[category]["total"] += 1
        if item["refused"]:
            categories[category]["refused"] += 1

    # Calculate ASR for each category
    for category, data in categories.items():
        data["asr"] = (data["total"] - data["refused"]) / data["total"]
        data["refusal_rate"] = data["refused"] / data["total"]

    return categories


def print_comparison(baseline_results: Dict, lora_results: Dict):
    """Print comparison between baseline and LoRA models."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':<20} {'LoRA':<20} {'Change':<10}")
    print("-" * 70)

    baseline_asr = baseline_results["asr"]
    lora_asr = lora_results["asr"]
    change = lora_asr - baseline_asr

    print(f"{'ASR (Attack Success Rate)':<25} {baseline_asr:>18.1%} {lora_asr:>18.1%} {change:>+9.1%}")
    print(f"{'Refusal Rate':<25} {baseline_results['refusal_rate']:>18.1%} {lora_results['refusal_rate']:>18.1%} {-change:>+9.1%}")
    print(f"{'Refusals':<25} {baseline_results['refused']:>20d} {lora_results['refused']:>20d} {lora_results['refused'] - baseline_results['refused']:>+10d}")
    print(f"{'Complied':<25} {baseline_results['complied']:>20d} {lora_results['complied']:>20d} {lora_results['complied'] - baseline_results['complied']:>+10d}")

    print("=" * 70)

    # Category breakdown
    print("\nCATEGORY BREAKDOWN:")
    print("-" * 70)
    print(f"{'Category':<25} {'Baseline ASR':<15} {'LoRA ASR':<15} {'Change':<10}")
    print("-" * 70)

    baseline_cats = evaluate_by_category(baseline_results["results"])
    lora_cats = evaluate_by_category(lora_results["results"])

    for category in baseline_cats:
        b_asr = baseline_cats[category]["asr"]
        l_asr = lora_cats.get(category, {}).get("asr", 0)
        change = l_asr - b_asr
        print(f"{category:<25} {b_asr:>13.1%} {l_asr:>13.1%} {change:>+9.1%}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA fine-tuned model")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Compare with baseline model (no LoRA)")
    parser.add_argument("--baseline-results", type=str,
                        help="Path to baseline results JSON (skip baseline eval)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file for results")
    args = parser.parse_args()

    # Load prompts
    prompts_file = Path(__file__).resolve().parent.parent / "data" / "dangerous_prompts.json"
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    print(f"Loaded {len(prompts_data)} dangerous prompts")

    # Load models
    lora_model, lora_tokenizer = load_model_with_lora()
    if lora_model is None:
        print("Failed to load LoRA model. Exiting.")
        return

    # Evaluate baseline if requested
    baseline_results = None
    if args.baseline_results and Path(args.baseline_results).exists():
        print(f"Loading baseline results from {args.baseline_results}...")
        with open(args.baseline_results, "r") as f:
            baseline_results = json.load(f)
    elif args.compare_baseline:
        baseline_model, baseline_tokenizer = load_baseline_model()
        baseline_results = evaluate_model(baseline_model, baseline_tokenizer, prompts_data, "Baseline")

    # Evaluate LoRA model
    lora_results = evaluate_model(lora_model, lora_tokenizer, prompts_data, "LoRA (Layers 9-10)")

    # Compare if baseline results available
    if baseline_results:
        print_comparison(baseline_results, lora_results)

    # Save results
    output_file = OUTPUT_DIR / args.output
    output_data = {
        "lora_results": lora_results,
    }
    if baseline_results:
        output_data["baseline_results"] = baseline_results
        output_data["comparison"] = {
            "baseline_asr": baseline_results["asr"],
            "lora_asr": lora_results["asr"],
            "asr_improvement": lora_results["asr"] - baseline_results["asr"],
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
