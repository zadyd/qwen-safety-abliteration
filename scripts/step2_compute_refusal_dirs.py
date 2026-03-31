"""
Step 2: Compute the refusal direction for each layer.

Using the activations extracted in Step 1:
  - For each of the 24 layers, compute:
      refusal_direction = mean(last_token_harmful) - mean(last_token_safe)
  - Normalize to unit vector
  - Save all directions to results/refusal_directions.pt

Output:
  - results/refusal_directions.pt  (Dict[layer_idx, Tensor[hidden_dim]])
  - Console: per-layer contribution scores
"""
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layer_analysis import (
    compute_refusal_direction,
    compute_all_refusal_directions,
    compute_layer_contribution_score,
    compute_refusal_direction_from_mlp,
)
from configs.config import (
    ACTIVATIONS_DIR, REFUSAL_DIRS_FILE, RESULTS_DIR,
    NUM_LAYERS, FULL_ATTENTION_LAYERS
)


def main():
    harmful_path = ACTIVATIONS_DIR / "harmful_activations.pt"
    safe_path = ACTIVATIONS_DIR / "safe_activations.pt"

    print("Loading activations...")
    harmful_activations = torch.load(harmful_path, map_location="cpu")
    safe_activations = torch.load(safe_path, map_location="cpu")

    print(f"Loaded activations for {len(harmful_activations)} layers")

    # Try multiple methods for residual-stream directions
    methods = ["last_token", "mean_pool", "max_pool"]
    refusal_directions_best = None
    best_scores = None

    for method in methods:
        print(f"\nComputing refusal directions (method={method})...")
        dirs = compute_all_refusal_directions(
            harmful_activations,
            safe_activations,
            method=method
        )
        scores = {}
        for layer_idx in sorted(dirs.keys()):
            direction = dirs[layer_idx]
            h_act = harmful_activations[layer_idx]
            s_act = safe_activations[layer_idx]
            scores[layer_idx] = compute_layer_contribution_score(direction, h_act, s_act)

        total_score = sum(abs(v) for v in scores.values())
        print(f"  Total |projection| across layers: {total_score:.4f}")
        if refusal_directions_best is None or total_score > sum(abs(v) for v in best_scores.values()):
            refusal_directions_best = dirs
            best_scores = scores
            best_method = method

    refusal_directions = refusal_directions_best
    scores = best_scores
    print(f"\nBest method: {best_method} (highest total projection score)")

    # Also try MLP-based directions if available
    mlp_h_path = ACTIVATIONS_DIR / "harmful_mlp_activations.pt"
    mlp_s_path = ACTIVATIONS_DIR / "safe_mlp_activations.pt"
    mlp_dirs = None
    if mlp_h_path.exists() and mlp_s_path.exists():
        print("\nLoading MLP activations...")
        harmful_mlp = torch.load(mlp_h_path, map_location="cpu")
        safe_mlp = torch.load(mlp_s_path, map_location="cpu")
        print(f"Loaded MLP activations for {len(harmful_mlp)} layers")

        mlp_dirs = compute_refusal_direction_from_mlp(harmful_mlp, safe_mlp, method="last_token")
        mlp_scores = {}
        for layer_idx in mlp_dirs:
            direction = mlp_dirs[layer_idx]
            h_act = harmful_mlp[layer_idx]
            s_act = safe_mlp[layer_idx]
            # Per-pair projection then average — handles any seq length mismatch
            n = h_act.shape[0]
            proj_scores = []
            for i in range(n):
                h_f = h_act[i, -1, :].mean(dim=0)
                s_f = s_act[i, -1, :].mean(dim=0)
                proj_scores.append(torch.dot(h_f - s_f, direction).item())
            mlp_scores[layer_idx] = sum(proj_scores) / len(proj_scores)

        mlp_total = sum(abs(v) for v in mlp_scores.values())
        print(f"MLP total |projection| score: {mlp_total:.4f}")

        # If MLP has stronger signal, prefer it
        residual_total = sum(abs(v) for v in scores.values())
        if mlp_total > residual_total * 1.5:
            print("MLP directions have significantly stronger signal — using MLP directions")
            refusal_directions = mlp_dirs
            scores = mlp_scores
        else:
            print("Using residual-stream directions (MLP signal not strong enough to override)")

    # Print per-layer scores
    print("\nPer-layer refusal signal strength (projection score):")
    print("-" * 55)
    print(f"{'Layer':>6} | {'Attention Type':>16} | {'Score':>10}")
    print("-" * 55)

    for layer_idx in sorted(scores.keys()):
        direction = refusal_directions[layer_idx]
        score = scores[layer_idx]
        attn_type = "FULL_ATTN" if layer_idx in FULL_ATTENTION_LAYERS else "LINEAR_ATTN"
        print(f"  {layer_idx:2d}   | {attn_type:16s} | {score:10.4f}")

    print("-" * 55)

    sorted_layers = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\nTop 6 layers by refusal signal strength:")
    for rank, (layer_idx, score) in enumerate(sorted_layers[:6], 1):
        attn_type = "FULL" if layer_idx in FULL_ATTENTION_LAYERS else "LINEAR"
        print(f"  #{rank} Layer {layer_idx} ({attn_type}): {score:.4f}")

    # Save refusal directions
    REFUSAL_DIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(refusal_directions, REFUSAL_DIRS_FILE)
    print(f"\nSaved refusal directions to: {REFUSAL_DIRS_FILE}")

    # Save scores as JSON for step 4
    scores_path = RESULTS_DIR / "refusal_scores.json"
    with open(scores_path, "w") as f:
        json.dump({str(k): float(v) for k, v in scores.items()}, f, indent=2)
    print(f"Saved refusal scores to: {scores_path}")

    print("\nStep 2 complete!")


if __name__ == "__main__":
    main()
