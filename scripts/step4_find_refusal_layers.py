"""
Step 4: Analyze results and generate visualizations.

Reads:
  - results/abliteration_results.json  (per-layer Abliteration results)
  - results/refusal_scores.json         (per-layer refusal direction scores)

Outputs:
  - results/per_layer_refusal_drop.png   (bar chart)
  - results/findings.md                  (summary report)
"""
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    ABLITERATION_RESULTS_FILE, RESULTS_DIR,
    NUM_LAYERS, FULL_ATTENTION_LAYERS
)


def plot_per_layer_refusal_drop(results: dict, output_path: Path):
    """
    Generate a bar chart showing refusal rate drop for each layer.
    Bars are colored by attention type: blue = linear, red = full.
    """
    layers = list(range(NUM_LAYERS))
    refusal_drops = []
    colors = []

    for layer_idx in layers:
        data = results["per_layer"].get(str(layer_idx), results["per_layer"].get(layer_idx, {}))
        drop = data.get("refusal_drop", 0)
        if drop is None:
            drop = 0.0
        refusal_drops.append(drop)
        colors.append("indianred" if layer_idx in FULL_ATTENTION_LAYERS else "steelblue")

    fig, ax = plt.subplots(figsize=(14, 6))

    bar_locs = list(range(NUM_LAYERS))
    bars = ax.bar(bar_locs, refusal_drops, color=colors, width=0.8, alpha=0.85, edgecolor="white")

    for bar, drop in zip(bars, refusal_drops):
        if abs(drop) > 0.03:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{drop:+.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="dimgray",
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    for i in range(4, NUM_LAYERS, 4):
        ax.axvline(i - 0.5, color="lightgray", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Refusal Rate Drop (+ means more jailbroken)", fontsize=12)
    ax.set_title("Qwen3.5-2B Abliteration: Per-Layer Refusal Signal Contribution", fontsize=13, fontweight="bold")
    ax.set_xticks(range(NUM_LAYERS))
    ax.set_xticklabels([str(i) for i in range(NUM_LAYERS)], fontsize=8)
    ax.set_ylim(-0.15, max(refusal_drops) * 1.15 + 0.05 if refusal_drops else 0.5)

    full_layers_str = ",".join(map(str, FULL_ATTENTION_LAYERS))
    full_patch = mpatches.Patch(color="indianred", label=f"Full Attention (layers {full_layers_str})")
    linear_patch = mpatches.Patch(color="steelblue", label="Linear Attention (other layers)")
    ax.legend(handles=[full_patch, linear_patch], loc="upper right", fontsize=9)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved bar chart to: {output_path}")


def plot_heatmap(results: dict, output_path: Path):
    """
    Generate a heatmap showing refusal drop across layers and prompt categories.
    """
    per_layer = results["per_layer"]

    # Build a simple heatmap array: [num_layers x 1]
    drops = []
    for layer_idx in range(NUM_LAYERS):
        drop = per_layer.get(str(layer_idx), {}).get("refusal_drop", 0)
        if drop is None:
            drop = 0.0
        drops.append(drop)

    # Reshape into a 4-row grid (6 columns)
    rows = 4
    cols = 6
    grid = []
    for r in range(rows):
        row = drops[r * cols: (r + 1) * cols]
        grid.append(row)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=-0.2, vmax=0.5)

    # Labels
    ax.set_xticks([i for i in range(0, NUM_LAYERS, cols)])
    ax.set_xticklabels([str(i) for i in range(0, NUM_LAYERS, cols)], fontsize=9)
    ax.set_yticks(range(rows))
    layer_labels = [f"Layers {r*cols}-{(r+1)*cols-1}" for r in range(rows)]
    ax.set_yticklabels(layer_labels, fontsize=9)
    ax.set_xlabel("Layer Within Group", fontsize=11)
    ax.set_title("Qwen3.5-2B Refusal Signal Heatmap", fontsize=12, fontweight="bold")

    # Annotate cells
    for r in range(rows):
        for c in range(cols):
            layer_idx = r * cols + c
            val = grid[r][c]
            color = "white" if abs(val) > 0.25 else "black"
            ax.text(c, r, f"L{layer_idx}\n{val:+.0%}",
                    ha="center", va="center", fontsize=7.5, color=color)

    plt.colorbar(im, ax=ax, label="Refusal Drop (+ = more jailbroken)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def plot_batch_group_results(results: dict, output_path: Path):
    """Generate a bar chart comparing batch abliteration groups."""
    batch_groups = results.get("batch_groups", {})
    if not batch_groups:
        return

    group_names = []
    refusal_drops = []
    colors = []

    for gn, gd in sorted(batch_groups.items(), key=lambda x: x[1].get("refusal_drop", 0), reverse=True):
        group_names.append(gn)
        drop = gd.get("refusal_drop", 0) or 0
        refusal_drops.append(drop)
        colors.append("crimson" if drop > 0.1 else "steelblue")

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(group_names, refusal_drops, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Refusal Rate Drop (+ means more jailbroken)", fontsize=11)
    ax.set_ylabel("Layer Group", fontsize=11)
    ax.set_title("Qwen3.5-2B Batch Abliteration: Group-Level Refusal Disruption", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, drop in zip(bars, refusal_drops):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{drop:+.1%}", va="center", fontsize=9, color="dimgray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved batch-group chart to: {output_path}")


def generate_findings_markdown(results: dict, refusal_scores: dict, output_path: Path):
    baseline_refusal = results["baseline_refusal_rate"]
    per_layer = results["per_layer"]
    batch_groups = results.get("batch_groups", {})

    # Sort layers by refusal drop
    sorted_layers = sorted(
        [(int(k), v) for k, v in per_layer.items()],
        key=lambda x: abs(x[1].get("refusal_drop", 0) or 0),
        reverse=True
    )

    critical_layers = [
        (layer_idx, data) for layer_idx, data in sorted_layers
        if (data.get("refusal_drop", 0) or 0) > 0.05
    ]

    # Find best batch group
    best_group = None
    best_drop = -999
    for group_name, gdata in batch_groups.items():
        drop = gdata.get("refusal_drop", 0) or 0
        if drop > best_drop:
            best_drop = drop
            best_group = (group_name, gdata)

    lines = [
        "# Qwen3.5-2B Abliteration Findings",
        "",
        f"## Baseline",
        f"- Baseline refusal rate: {baseline_refusal:.1%}",
        f"- Baseline ASR: {results['baseline_asr']:.1%}",
        "",
        f"## Key Findings",
        "",
        f"### Batch Abliteration Results (multi-layer groups)",
    ]

    if batch_groups:
        sorted_groups = sorted(batch_groups.items(), key=lambda x: x[1].get("refusal_drop", 0) or 0, reverse=True)
        lines.extend([
            "",
            "| Group        | Layers               | ASR    | Refusal Drop |",
            "|--------------|----------------------|--------|--------------|",
        ])
        for group_name, gdata in sorted_groups:
            layers_str = str(gdata.get("layers", []))
            asr = gdata.get("asr", 0) or 0
            drop = gdata.get("refusal_drop", 0) or 0
            lines.append(f"| {group_name:13s} | {layers_str:20s} | {asr:5.1%} | {drop:+11.1%} |")
        lines.append("")

        if best_group:
            gn, gd = best_group
            lines.extend([
                f"**Most effective group**: `{gn}` — ASR={gd.get('asr', 0):.1%}, "
                f"refusal_drop={gd.get('refusal_drop', 0):+.1%}",
                f"Active layers: {gd.get('active_layers', [])}",
                "",
            ])
    else:
        lines.append("No batch groups evaluated.")

    lines.extend([
        f"### Critical Refusal Layers (refusal drop > 5%)",
    ])

    if critical_layers:
        lines.append("")
        for layer_idx, data in critical_layers:
            attn = "FULL_ATTN" if layer_idx in FULL_ATTENTION_LAYERS else "LINEAR_ATTN"
            drop = data.get("refusal_drop", 0) or 0
            lines.append(f"- **Layer {layer_idx}** ({attn}): refusal drop = {drop:+.1%}")
    else:
        lines.append("No layers showed >5% refusal drop individually.")

    lines.extend([
        "",
        f"### Full Attention Layers (n={len(FULL_ATTENTION_LAYERS)})",
        f"Indices: {FULL_ATTENTION_LAYERS}",
        "",
        f"### Linear Attention Layers (n={NUM_LAYERS - len(FULL_ATTENTION_LAYERS)})",
        "",
        "## Per-Layer Detailed Results",
        "",
        "| Layer | Type       | ASR     | Refusal Drop |",
        "|-------|------------|---------|--------------|",
    ])

    for layer_idx in range(NUM_LAYERS):
        data = per_layer.get(str(layer_idx), per_layer.get(layer_idx, {}))
        attn_type = data.get("attention_type", "N/A")
        asr = data.get("asr")
        drop = data.get("refusal_drop")
        asr_str = f"{asr:.1%}" if asr is not None else "N/A"
        drop_str = f"{drop:+.1%}" if drop is not None else "N/A"
        lines.append(f"| {layer_idx:5d} | {attn_type:10s} | {asr_str:>7s} | {drop_str:>12s} |")

    lines.extend([
        "",
        "## Top Layers by Refusal Direction Score",
        "",
        "| Rank | Layer | Score |",
        "|------|-------|-------|",
    ])

    sorted_scores = sorted(refusal_scores.items(), key=lambda x: abs(float(x[1])), reverse=True)
    for rank, (layer_idx, score) in enumerate(sorted_scores[:10], 1):
        lines.append(f"| {rank:4d} | {layer_idx:5s} | {float(score):6.4f} |")

    lines.extend([
        "",
        "## Conclusions",
        "",
        "- TBD: analyze which attention types dominate the refusal signal",
        "- TBD: identify whether refusal is concentrated in later layers",
        "- TBD: compare with Qwen3.5-27B findings (layers 48-63)",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved findings report to: {output_path}")


def main():
    # Load results
    with open(ABLITERATION_RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    scores_path = RESULTS_DIR / "refusal_scores.json"
    with open(scores_path, "r") as f:
        refusal_scores = json.load(f)

    print("Generating visualizations...")

    # Bar chart
    plot_per_layer_refusal_drop(results, RESULTS_DIR / "per_layer_refusal_drop.png")

    # Heatmap
    plot_heatmap(results, RESULTS_DIR / "refusal_signal_heatmap.png")

    # Batch-group comparison chart
    plot_batch_group_results(results, RESULTS_DIR / "batch_group_refusal_drop.png")

    # Findings report
    generate_findings_markdown(results, refusal_scores, RESULTS_DIR / "findings.md")

    print("\nStep 4 complete!")
    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
