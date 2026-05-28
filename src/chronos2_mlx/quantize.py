"""Weight-only quantization (int4/int8) and QLoRA support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class QuantizeConfig:
    bits: int = 8
    group_size: int = 64


def _linear_dims(module: nn.Module) -> tuple[int, int]:
    """Return (out_dim, in_dim) for both Linear and QuantizedLinear."""
    if isinstance(module, nn.QuantizedLinear):
        out_dim = module.scales.shape[0]
        in_dim = module.scales.shape[1] * module.group_size
        return out_dim, in_dim
    return module.weight.shape  # (out_dim, in_dim)


def _should_quantize(path: str, module: nn.Module, group_size: int) -> bool | dict:
    """Predicate for nn.quantize — skips embeddings, LoRA wrappers, and non-divisible layers."""
    from .adapters import LoRALinear
    if not isinstance(module, nn.Linear):
        return False
    if isinstance(module, LoRALinear):
        return False
    out_dim, in_dim = module.weight.shape
    # MLX requires in_dim % group_size == 0
    if in_dim < group_size or in_dim % group_size != 0:
        return False
    return True


def quantize_model(
    model: nn.Module,
    config: QuantizeConfig | None = None,
    *,
    bits: int | None = None,
    group_size: int | None = None,
) -> dict:
    """
    Quantize all eligible nn.Linear layers in a Chronos-2 MLX model.

    Skips embedding tables and layers smaller than group_size.
    Returns a stats dict with {num_quantized, bits, group_size}.
    """
    if config is None:
        config = QuantizeConfig(
            bits=bits if bits is not None else 8,
            group_size=group_size if group_size is not None else 64,
        )

    quantized: list[str] = []

    def predicate(path: str, module: nn.Module):
        if not _should_quantize(path, module, config.group_size):
            return False
        quantized.append(path)
        return {"group_size": config.group_size, "bits": config.bits}  # type: ignore[return-value]

    nn.quantize(model, class_predicate=predicate)
    mx.eval(model.parameters())

    return {"num_quantized": len(quantized), "bits": config.bits, "group_size": config.group_size}


def dequantize_model(model: nn.Module) -> int:
    """Replace all QuantizedLinear layers back to regular Linear (for fusing/export)."""
    from .adapters import _get_module, _set_module

    count = 0
    for path, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            out_dim, in_dim = _linear_dims(module)
            bias = module.bias if hasattr(module, "bias") and module.bias is not None else None
            lin = nn.Linear(in_dim, out_dim, bias=bias is not None)
            # Dequantize weight
            lin.weight = mx.dequantize(
                module.weight,
                module.scales,
                module.biases,
                module.group_size,
                module.bits,
            )
            if bias is not None:
                lin.bias = bias
            _set_module_by_path(model, path, lin)
            count += 1

    mx.eval(model.parameters())
    return count


def _set_module_by_path(model: nn.Module, path: str, value: nn.Module) -> None:
    """Navigate dotted path and set the final attribute."""
    if not path:
        return
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


def param_footprint(model: nn.Module) -> dict:
    """Return total params and effective bits (accounting for QuantizedLinear)."""
    total_bits = 0
    total_params = 0
    quantized_params = 0

    for path, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            out_dim, in_dim = _linear_dims(module)
            n = out_dim * in_dim
            total_params += n
            quantized_params += n
            # Quantized bits for weights + fp32 overhead for scales/biases
            scale_overhead = module.scales.size + module.biases.size
            total_bits += n * module.bits + scale_overhead * 32
        elif isinstance(module, nn.Linear):
            n = module.weight.size
            total_params += n
            total_bits += n * 32  # assume fp32 baseline
        elif isinstance(module, nn.Embedding):
            n = module.weight.size
            total_params += n
            total_bits += n * 32

    return {
        "total_params": total_params,
        "quantized_params": quantized_params,
        "total_bytes": total_bits // 8,
    }
