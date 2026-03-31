"""
Layer-wise activation analysis and refusal direction computation.
"""
import torch
from typing import Dict, List, Literal
def compute_refusal_direction(
    harmful_activation: torch.Tensor,
    safe_activation: torch.Tensor,
    method: Literal["last_token", "mean_pool", "max_pool"] = "last_token"
) -> torch.Tensor:
    """
    Compute the refusal direction for a single layer.

    Args:
        harmful_activation: [batch, seq_len_h, hidden_dim] - activation for harmful prompts
        safe_activation:   [batch, seq_len_s, hidden_dim] - activation for safe prompts
                          batch must match, but seq_len_h and seq_len_s may differ
                          (each row i corresponds to the i-th prompt pair).
        method:
            - 'last_token': takes the last position of each prompt pair (handles
              different harmful/safe seq lengths naturally — no padding needed)
            - 'mean_pool': averages each prompt independently, then subtracts
            - 'max_pool':  max-over-sequence per prompt

    Returns:
        refusal_direction: [hidden_dim] normalized vector
    """
    # Handle 4D tensors from KV-cache extraction: [batch, num_kv_heads, seq, hidden]
    # Squeeze out the num_kv_heads dimension (typically 1 for these models).
    if harmful_activation.dim() == 4:
        harmful_activation = harmful_activation.squeeze(1)
    if safe_activation.dim() == 4:
        safe_activation = safe_activation.squeeze(1)

    n = harmful_activation.shape[0]
    assert safe_activation.shape[0] == n, \
        f"Batch size mismatch: harmful={n}, safe={safe_activation.shape[0]}"

    directions = []
    for i in range(n):
        h_i = harmful_activation[i]   # [seq_h, hidden]
        s_i = safe_activation[i]      # [seq_s, hidden]

        if method == "last_token":
            h_feat = h_i[-1, :]        # last token of harmful prompt
            s_feat = s_i[-1, :]        # last token of safe prompt
        elif method == "mean_pool":
            h_feat = h_i.mean(dim=0)
            s_feat = s_i.mean(dim=0)
        elif method == "max_pool":
            h_feat = h_i.max(dim=0).values
            s_feat = s_i.max(dim=0).values
        else:
            raise ValueError(f"Unknown method: {method}")

        directions.append(h_feat - s_feat)

    refusal_dir = torch.stack(directions).mean(dim=0)   # [hidden]
    refusal_dir = refusal_dir / refusal_dir.norm()
    return refusal_dir


def compute_all_refusal_directions(
    harmful_activations: Dict[int, torch.Tensor],
    safe_activations: Dict[int, torch.Tensor],
    method: Literal["last_token", "mean_pool"] = "last_token"
) -> Dict[int, torch.Tensor]:
    """
    Compute refusal directions for all layers.

    Args:
        harmful_activations: Dict[layer_idx -> activation tensor]
        safe_activations: Dict[layer_idx -> activation tensor]
        method: 'last_token' or 'mean_pool'

    Returns:
        refusal_directions: Dict[layer_idx -> direction tensor]
    """
    refusal_directions = {}
    for layer_idx in harmful_activations:
        refusal_directions[layer_idx] = compute_refusal_direction(
            harmful_activations[layer_idx],
            safe_activations[layer_idx],
            method=method
        )
    return refusal_directions


def compute_layer_contribution_score(
    refusal_direction: torch.Tensor,
    harmful_activation: torch.Tensor,
    safe_activation: torch.Tensor
) -> float:
    """
    Compute a scalar score indicating how strongly this layer
    contributes to refusal behavior.

    Measures the projection of harmful activation onto the refusal direction
    minus the projection of safe activation onto that direction.
    Per-pair projection, then averaged — works even when harmful/safe
    have different sequence lengths.
    """
    n = harmful_activation.shape[0]
    assert safe_activation.shape[0] == n
    scores = []
    for i in range(n):
        h_last = harmful_activation[i, -1, :].mean(dim=0)
        s_last = safe_activation[i, -1, :].mean(dim=0)
        scores.append(torch.dot(h_last, refusal_direction).item()
                      - torch.dot(s_last, refusal_direction).item())
    return sum(scores) / len(scores)


def compute_refusal_direction_from_mlp(
    harmful_mlp_activations: Dict[int, torch.Tensor],
    safe_mlp_activations: Dict[int, torch.Tensor],
    method: Literal["last_token", "mean_pool", "max_pool"] = "last_token"
) -> Dict[int, torch.Tensor]:
    """
    Compute refusal directions from MLP-layer (down_proj output) activations
    instead of residual-stream output.

    The MLP activation is computed after the activation function and
    represents what the feedforward sub-layer *chooses* to contribute —
    more directly tied to the layer's internal computation.

    Args:
        harmful_mlp_activations: Dict[layer_idx -> Tensor[batch, seq, intermediate_dim]]
        safe_mlp_activations:   Dict[layer_idx -> Tensor[batch, seq, intermediate_dim]]
        method: 'last_token', 'mean_pool', or 'max_pool'

    Returns:
        refusal_directions: Dict[layer_idx -> Tensor[intermediate_dim]]
    """
    refusal_directions = {}
    for layer_idx in harmful_mlp_activations:
        if layer_idx not in safe_mlp_activations:
            continue
        h_act = harmful_mlp_activations[layer_idx]
        s_act = safe_mlp_activations[layer_idx]

        # Handle 4D tensors: [batch, num_heads, seq, intermediate] -> squeeze to 3D
        if h_act.dim() == 4:
            h_act = h_act.squeeze(1)
        if s_act.dim() == 4:
            s_act = s_act.squeeze(1)

        n = h_act.shape[0]
        directions = []
        for i in range(n):
            h_i = h_act[i]   # [seq_h, intermediate]
            s_i = s_act[i]   # [seq_s, intermediate]

            if method == "last_token":
                h_f = h_i[-1, :]
                s_f = s_i[-1, :]
            elif method == "mean_pool":
                h_f = h_i.mean(dim=0)
                s_f = s_i.mean(dim=0)
            elif method == "max_pool":
                h_f = h_i.max(dim=0).values
                s_f = s_i.max(dim=0).values
            else:
                raise ValueError(f"Unknown method: {method}")

            directions.append(h_f - s_f)

        refusal_dir = torch.stack(directions).mean(dim=0)
        if refusal_dir.norm() > 1e-8:
            refusal_dir = refusal_dir / refusal_dir.norm()
        refusal_directions[layer_idx] = refusal_dir

    return refusal_directions
