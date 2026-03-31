"""
PyTorch forward hook utilities for activation extraction.
"""
import json
import torch
from typing import Dict, List, Callable, Any
import pathlib as _pathlib, datetime as _datetime


_log_path = _pathlib.Path(r"C:\Users\ZADYD\Desktop\qwen_safety_test\debug-d3b09c.log")

def _ndjson_log(sessionId, runId, hypothesisId, location, message, data):
    entry = json.dumps({
        "sessionId": sessionId,
        "id": f"log_{_datetime.datetime.now().timestamp():.0f}_{hypothesisId}",
        "timestamp": int(_datetime.datetime.now().timestamp() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "runId": runId,
        "hypothesisId": hypothesisId
    })
    print(f"[DBG] {message} | {data}", flush=True)
    try:
        with open(_log_path, "a") as _lf:
            _lf.write(entry + "\n")
    except Exception:
        pass


class ActivationExtractor:

    def __init__(self, model, target_layers: List[int]):
        self.hooks: List[Any] = []
        self.activations: Dict[int, torch.Tensor] = {}
        # Log model structure
        _ndjson_log("d3b09c", "debug", "H1",
            "hooks.py:__init__",
            "model_named_children",
            {n: type(getattr(model, n, None)).__name__ for n, _ in model.named_children()}
        )
        self._register_hooks(model, target_layers)

    @staticmethod
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
            ("language_model", "model", "layers"),            # 3-level
            ("model", "model", "layers"),                     # 3-level
            ("model", "layers"),                              # 2-level
            ("language_model", "layers"),                      # 2-level
            ("transformer", "h"),                              # 1-level
            ("transformer", "blocks"),                         # 1-level
        ]
        _ndjson_log("d3b09c", "debug", "H2",
            "hooks.py:_find_layer_parent",
            "trying_candidates",
            [{"pattern": c} for c in candidates]
        )

        def get_by_chain(obj, attrs):
            """Safely follow an attribute chain."""
            for attr in attrs:
                if obj is None:
                    return None
                obj = getattr(obj, attr, None)
            return obj

        for chain in candidates:
            layers = get_by_chain(model, chain)
            if isinstance(layers, (list, torch.nn.ModuleList)) and len(layers) > 0:
                path_str = ".".join(chain)
                _ndjson_log("d3b09c", "debug", "H2",
                    "hooks.py:_find_layer_parent",
                    "found_layers",
                    {"path": path_str, "num_layers": len(layers), "layer_type": type(layers[0]).__name__}
                )
                return layers

        _ndjson_log("d3b09c", "debug", "H3",
            "hooks.py:_find_layer_parent",
            "NOT_FOUND",
            {"model_class": type(model).__name__}
        )
        raise RuntimeError(
            f"Could not find layers in model. "
            f"Model type: {type(model).__name__}. "
            f"Please check model structure and update _find_layer_parent()."
        )

    def _register_hooks(self, model, target_layers: List[int]):
        layers = self._find_layer_parent(model)
        _ndjson_log("d3b09c", "debug", "H1",
            "hooks.py:_register_hooks",
            "registering_hooks",
            {"num_layers": len(layers), "target": target_layers}
        )
        for i in target_layers:
            layer = layers[i]
            handle = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(handle)

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.activations[layer_idx] = output[0].detach().clone()
            else:
                self.activations[layer_idx] = output.detach().clone()
        return hook

    def clear_data(self):
        """Clear activation data dicts without removing hooks."""
        self.activations.clear()
        if hasattr(self, "mlp_activations"):
            self.mlp_activations.clear()

    def clear(self):
        """Remove all hooks and clear all data."""
        for h in self.hooks:
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass
        self.hooks.clear()
        self.activations.clear()
        if hasattr(self, "mlp_activations"):
            self.mlp_activations.clear()

    def get_activation(self, layer_idx: int) -> torch.Tensor:
        return self.activations.get(layer_idx, None)

    def get_all_activations(self) -> Dict[int, torch.Tensor]:
        return self.activations.copy()


class DualActivationExtractor(ActivationExtractor):
    """
    Extends ActivationExtractor with a second set of hooks that capture
    the MLP down_proj output of each Transformer layer.

    The MLP activation is the output of the feedforward sub-layer — it
    represents what each layer *chooses* to contribute to the residual
    stream, providing a richer refusal-direction signal than the
    residual-stream output alone.
    """

    def __init__(self, model, target_layers: List[int]):
        self.mlp_activations: Dict[int, torch.Tensor] = {}
        super().__init__(model, target_layers)
        self._register_mlp_hooks(model, target_layers)

    def _register_mlp_hooks(self, model, target_layers: List[int]):
        layers = self._find_layer_parent(model)
        _ndjson_log("d3b09c", "debug", "H4",
            "hooks.py:_register_mlp_hooks",
            "scanning_mlp",
            {"num_target_layers": len(target_layers)}
        )
        for i in target_layers:
            layer = layers[i]
            mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
            if mlp is not None:
                down_proj = getattr(mlp, "down_proj", None)
                if down_proj is not None:
                    h = down_proj.register_forward_hook(self._make_mlp_hook(i))
                    self.hooks.append(h)
                    _ndjson_log("d3b09c", "debug", "H4",
                        "hooks.py:_register_mlp_hooks",
                        "registered_mlp_hook",
                        {"layer": i, "down_proj_shape": list(down_proj.weight.shape)}
                    )

    def _make_mlp_hook(self, layer_idx: int) -> Callable:
        def hook(module, input, output):
            self.mlp_activations[layer_idx] = output.detach().clone()
        return hook

    def get_mlp_activations(self) -> Dict[int, torch.Tensor]:
        return self.mlp_activations.copy()
