"""
Core Abliteration logic: inject hooks to neutralize refusal directions.
"""
import torch
from typing import List, Dict, Literal


def _find_layer_parent(model):
    """
    Traverse any depth of attribute chain to find the ModuleList of
    Transformer layers. Tries from deepest (4-level) to shallowest (1-level).

    Supported paths for Qwen multimodal models:
      - 4-level: model.language_model.model.layers  (Qwen3_5ForConditionalGeneration)
      - 3-level: model.model.layers  /  language_model.model.layers
      - 2-level: model.layers  /  language_model.layers
      - 1-level: transformer.h  /  transformer.blocks
    """
    candidates = [
        ("model", "language_model", "model", "layers"),  # 4-level: multimodal
        ("model", "language_model", "layers"),            # 3-level
        ("language_model", "model", "layers"),             # 3-level
        ("model", "model", "layers"),                     # 3-level
        ("model", "layers"),                              # 2-level
        ("language_model", "layers"),                      # 2-level
        ("transformer", "h"),                              # 1-level
        ("transformer", "blocks"),                         # 1-level
    ]

    def get_by_chain(obj, attrs):
        for attr in attrs:
            if obj is None:
                return None
            obj = getattr(obj, attr, None)
        return obj

    for chain in candidates:
        layers = get_by_chain(model, chain)
        if isinstance(layers, (list, torch.nn.ModuleList)) and len(layers) > 0:
            return layers

    raise RuntimeError(
        f"Could not find layers in model. "
        f"Model type: {type(model).__name__}. "
        f"Please check model structure and update _find_layer_parent()."
    )


def abliterate_layer_output(
    model,
    layer_idx: int,
    refusal_direction: torch.Tensor,
    method: Literal["projection_nullify", "orthogonalize"] = "projection_nullify"
):
    """
    Inject a forward hook on a specific Transformer layer to neutralize
    the refusal direction in its output hidden_states.

    Args:
        model: The loaded Qwen3.5-2B model.
        layer_idx: Which Transformer layer to modify (0-23).
        refusal_direction: [hidden_dim] normalized direction vector.
        method:
            - 'projection_nullify': subtract the full projection onto refusal direction
            - 'orthogonalize': same as projection_nullify (alias)

    Returns:
        hook handle that can be used to remove the hook later.
    """
    # Pre-move direction to the device the layer will run on, so the hook
    # never incurs a CPU<->GPU transfer during inference.
    layer = _find_layer_parent(model)[layer_idx]
    layer_device = next(layer.parameters()).device
    layer_dtype = next(layer.parameters()).dtype
    refusal_dir = refusal_direction.to(device=layer_device, dtype=layer_dtype)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output

        if method in ("projection_nullify", "orthogonalize"):
            proj_score = torch.einsum("bSh,h->bS", hs, refusal_dir)
            hs = hs - proj_score.unsqueeze(-1) * refusal_dir

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    handle = layer.register_forward_hook(hook_fn)
    return handle


def abliterate_layers_batch(
    model,
    layer_indices: List[int],
    refusal_directions: Dict[int, torch.Tensor],
    method: Literal["projection_nullify", "orthogonalize"] = "projection_nullify"
):
    """
    Abliterate multiple layers at once (each with its own direction).

    Returns:
        List of hook handles (in order of layer_indices).
    """
    handles = []
    for layer_idx in layer_indices:
        direction = refusal_directions.get(layer_idx)
        if direction is not None:
            handle = abliterate_layer_output(model, layer_idx, direction, method)
            handles.append(handle)
        else:
            handles.append(None)
    return handles


def remove_hooks(handles: List):
    """Remove all hooks given their handles."""
    for h in handles:
        if h is not None:
            h.remove()
