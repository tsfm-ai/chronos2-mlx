"""LoRA adapters for Chronos-2 MLX fine-tuning."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    # Projections to adapt; layer_idx 0=time attention, 1=group attention
    target_projections: Sequence[str] = field(default_factory=lambda: ["q", "v"])
    target_attention_layers: Sequence[int] = field(default_factory=lambda: [0, 1])

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    # Legacy compat: if someone passes target_modules as a kwarg, parse it
    def __post_init__(self):
        pass


def _linear_dims(module: nn.Module) -> tuple[int, int]:
    """Return (out_dim, in_dim) for both nn.Linear and nn.QuantizedLinear."""
    if isinstance(module, nn.QuantizedLinear):
        out_dim = module.scales.shape[0]
        in_dim = module.scales.shape[1] * module.group_size
        return out_dim, in_dim
    return module.weight.shape  # (out_dim, in_dim)


class LoRALinear(nn.Module):
    """Drop-in wrapper for nn.Linear / nn.QuantizedLinear with low-rank adapter weights."""

    def __init__(self, base: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.freeze()

        out_dim, in_dim = _linear_dims(base)
        self.rank = rank
        self.scaling = alpha / rank

        # Initialize A with small normal, B with zeros (standard LoRA init)
        self.lora_a = mx.random.normal((rank, in_dim)) * 0.01
        self.lora_b = mx.zeros((out_dim, rank))

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        # delta = x @ A.T @ B.T  (applied in fp32 for stability)
        x32 = x.astype(mx.float32)
        a = self.lora_a.astype(mx.float32)
        b = self.lora_b.astype(mx.float32)
        delta = (x32 @ a.T) @ b.T
        if self.dropout is not None:
            delta = self.dropout(delta)
        return base_out + (self.scaling * delta).astype(base_out.dtype)

    def fuse(self) -> nn.Module:
        """Merge adapter into base weights.

        For QuantizedLinear: dequantizes, adds delta, re-quantizes.
        For Linear: adds delta in-place.
        """
        delta = self.scaling * (self.lora_b.astype(mx.float32) @ self.lora_a.astype(mx.float32))
        self.base.unfreeze()

        if isinstance(self.base, nn.QuantizedLinear):
            # Dequantize → add delta → re-quantize
            w = mx.dequantize(
                self.base.weight,
                self.base.scales,
                self.base.biases,
                self.base.group_size,
                self.base.bits,
            ).astype(mx.float32)
            w = w + delta
            out_dim, in_dim = _linear_dims(self.base)
            lin = nn.Linear(in_dim, out_dim, bias=False)
            lin.weight = w
            fused = nn.QuantizedLinear.from_linear(lin, self.base.group_size, self.base.bits)
            return fused

        self.base.weight = (self.base.weight.astype(mx.float32) + delta).astype(self.base.weight.dtype)
        return self.base

    def trainable_parameters(self):
        return {"lora_a": self.lora_a, "lora_b": self.lora_b}


def _get_module(model: nn.Module, path: str) -> nn.Module:
    """Navigate a dotted path like 'encoder.block.0.layer.0.time_attention.self_attention.q'."""
    parts = path.split(".")
    obj = model
    for p in parts:
        if isinstance(obj, list):
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def _set_module(model: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    obj = model
    for p in parts[:-1]:
        if isinstance(obj, list):
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    last = parts[-1]
    if isinstance(obj, list):
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def apply_lora(model, config: LoRAConfig) -> list[str]:
    """
    Inject LoRA adapters. Uses actual model paths:
      encoder.block.{i}.layer.{0 or 1}.self_attention.{q|k|v|o}
    layer.0 = TimeSelfAttention, layer.1 = GroupSelfAttention
    Returns list of paths where adapters were injected.
    """
    injected = []
    num_layers = model.config.num_layers

    for layer_idx in range(num_layers):
        for attn_idx in config.target_attention_layers:
            for proj in config.target_projections:
                module_path = f"encoder.block.{layer_idx}.layer.{attn_idx}.self_attention.{proj}"
                try:
                    base_linear = _get_module(model, module_path)
                    if not isinstance(base_linear, (nn.Linear, nn.QuantizedLinear)):
                        continue
                    lora_module = LoRALinear(
                        base_linear,
                        rank=config.rank,
                        alpha=config.alpha,
                        dropout=config.dropout,
                    )
                    _set_module(model, module_path, lora_module)
                    injected.append(module_path)
                except (AttributeError, IndexError, KeyError, ValueError):
                    continue

    return injected


def _all_lora_paths(model) -> list[str]:
    """Return all paths in the model that currently hold a LoRALinear."""
    paths = []
    num_layers = model.config.num_layers
    for layer_idx in range(num_layers):
        for attn_idx in range(2):
            for proj in ["q", "k", "v", "o"]:
                path = f"encoder.block.{layer_idx}.layer.{attn_idx}.self_attention.{proj}"
                try:
                    module = _get_module(model, path)
                    if isinstance(module, LoRALinear):
                        paths.append(path)
                except (AttributeError, IndexError, KeyError, ValueError):
                    continue
    return paths


def fuse_lora(model) -> None:
    """Fuse all LoRA adapters in place, returning model to base weights."""
    for path in _all_lora_paths(model):
        module = _get_module(model, path)
        fused = module.fuse()
        _set_module(model, path, fused)


def save_adapter(model, path: str | Path, config: LoRAConfig, base_model_id: str = "amazon/chronos-2") -> None:
    """Save LoRA adapter weights and config."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    adapter_weights = {}
    for weight_path in _all_lora_paths(model):
        module = _get_module(model, weight_path)
        adapter_weights[f"{weight_path}.lora_a"] = np.array(module.lora_a.astype(mx.float32))
        adapter_weights[f"{weight_path}.lora_b"] = np.array(module.lora_b.astype(mx.float32))

    np.savez(str(path / "adapter_model.npz"), **adapter_weights)

    meta = {
        "base_model": base_model_id,
        "adapter_type": "lora",
        "rank": config.rank,
        "alpha": config.alpha,
        "target_projections": list(config.target_projections),
        "target_attention_layers": list(config.target_attention_layers),
        "chronos2_mlx_version": "0.1.0",
    }
    (path / "adapter_config.json").write_text(json.dumps(meta, indent=2))


def load_adapter(model, path: str | Path, config: LoRAConfig | None = None) -> LoRAConfig:
    """Load LoRA adapter weights into model."""
    path = Path(path)

    meta = json.loads((path / "adapter_config.json").read_text())
    if config is None:
        config = LoRAConfig(
            rank=meta["rank"],
            alpha=meta["alpha"],
            target_projections=meta.get("target_projections", ["q", "v"]),
            target_attention_layers=meta.get("target_attention_layers", [0, 1]),
        )

    apply_lora(model, config)

    weights = dict(np.load(str(path / "adapter_model.npz")))

    for weight_path in _all_lora_paths(model):
        module = _get_module(model, weight_path)
        a_key = f"{weight_path}.lora_a"
        b_key = f"{weight_path}.lora_b"
        if a_key in weights:
            module.lora_a = mx.array(weights[a_key])
            module.lora_b = mx.array(weights[b_key])

    return config
