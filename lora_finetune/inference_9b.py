"""
Inference script for LoRA fine-tuned Qwen3.5-9B model.

Based on Step 3 ablation results (results_9b/abliteration_results.json):
  - Refusal signal is concentrated in EARLY layers (0-14), NOT the last layers.
  - Target layers for LoRA fine-tuning: layers 0-14.

Usage:
    python inference_9b.py --prompt "How can I make a bomb?"
    python inference_9b.py --interactive
    python inference_9b.py --test-dataset
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lora_finetune.config_9b import (
    MODEL_DIR,
    OUTPUT_DIR,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DEVICE,
    REFUSAL_KEYWORDS,
)


def load_model_with_lora():
    """Load the base model with LoRA adapter."""
    print(f"Loading base model from {MODEL_DIR}...")
    
    # Use 4-bit quantization to match training config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )

    adapter_path = OUTPUT_DIR / "final"
    if not adapter_path.exists():
        print(f"Warning: LoRA adapter not found at {adapter_path}")
        print("Using base model without LoRA...")
        return base_model

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )
    print(f"Model loaded successfully. Device: {next(model.parameters()).device}")
    return model


def load_tokenizer():
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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


def detect_refusal(response):
    """Detect if the response is a refusal."""
    response_lower = response.lower()
    for keyword in REFUSAL_KEYWORDS:
        if keyword.lower() in response_lower:
            return True
    return False


def test_single_prompt(model, tokenizer, prompt):
    """Test a single prompt and return the result."""
    print(f"\n{'=' * 60}")
    print(f"Prompt: {prompt}")
    print("-" * 60)

    response = generate_response(model, tokenizer, prompt)
    is_refused = detect_refusal(response)

    print(f"Response: {response}")
    print("-" * 60)
    print(f"Refusal detected: {is_refused}")

    return {
        "prompt": prompt,
        "response": response,
        "refused": is_refused,
    }


def test_dataset(model, tokenizer, prompts_file):
    """Test all prompts in the dataset."""
    print("\n" + "=" * 60)
    print("Testing all prompts in dataset...")
    print("=" * 60)

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompts = [item["harmful"] for item in prompts_data]
    results = []

    refused_count = 0
    for idx, prompt in enumerate(prompts):
        result = test_single_prompt(model, tokenizer, prompt)
        results.append(result)
        if result["refused"]:
            refused_count += 1

    total = len(prompts)
    asr = (total - refused_count) / total
    refusal_rate = refused_count / total

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total prompts: {total}")
    print(f"Refusals: {refused_count} ({refusal_rate:.1%})")
    print(f"ASR (Attack Success Rate): {asr:.1%}")
    print("=" * 60)

    return results


def interactive_mode(model, tokenizer):
    """Run in interactive mode."""
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'exit' to quit")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n> ")
            if prompt.lower() in ["exit", "quit", "q"]:
                break
            if not prompt.strip():
                continue

            test_single_prompt(model, tokenizer, prompt)
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(description="Inference with LoRA fine-tuned Qwen3.5-9B model")
    parser.add_argument("--prompt", type=str, help="Single prompt to test")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--test-dataset", action="store_true", help="Test all prompts in dataset")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature")
    parser.add_argument("--top-p", type=float, default=TOP_P, help="Top-p")

    args = parser.parse_args()

    model = load_model_with_lora()
    tokenizer = load_tokenizer()

    if isinstance(model, PeftModel):
        print("LoRA adapter loaded successfully!")
    else:
        print("Warning: Using base model without LoRA adapter")

    if args.prompt:
        test_single_prompt(model, tokenizer, args.prompt)
    elif args.interactive:
        interactive_mode(model, tokenizer)
    elif args.test_dataset:
        prompts_file = Path(__file__).resolve().parent.parent / "data" / "dangerous_prompts.json"
        results = test_dataset(model, tokenizer, prompts_file)
        results_file = OUTPUT_DIR / "inference_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {results_file}")
    else:
        print("Please specify --prompt, --interactive, or --test-dataset")
        parser.print_help()


if __name__ == "__main__":
    main()
