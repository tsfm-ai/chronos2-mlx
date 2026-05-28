"""Load official HuggingFace Chronos-2 safetensors weights into MLX model."""
from __future__ import annotations

import mlx.core as mx

from .model import Chronos2MLXModel


def _pt_to_mlx(tensor) -> mx.array:
    """Convert a PyTorch or numpy tensor to an MLX array."""
    import numpy as np
    if hasattr(tensor, "numpy"):
        return mx.array(tensor.numpy())
    return mx.array(np.array(tensor))


def load_chronos2_weights(model: Chronos2MLXModel, weights: dict) -> set[str]:
    """
    Load safetensors weights (as torch tensors or numpy arrays) into MLX model.
    Returns the set of weight keys that were consumed.
    """
    used = set()

    def get(key: str) -> mx.array:
        used.add(key)
        return _pt_to_mlx(weights[key])

    def load_rms_norm(module, prefix: str):
        module.weight = get(f"{prefix}.weight")

    def load_linear(module, prefix: str):
        module.weight = get(f"{prefix}.weight")
        bias_key = f"{prefix}.bias"
        if bias_key in weights:
            module.bias = get(bias_key)
            used.add(bias_key)

    def load_mha(module, prefix: str):
        load_linear(module.q, f"{prefix}.q")
        load_linear(module.k, f"{prefix}.k")
        load_linear(module.v, f"{prefix}.v")
        load_linear(module.o, f"{prefix}.o")

    def load_time_attn(module, prefix: str):
        load_rms_norm(module.layer_norm, f"{prefix}.layer_norm")
        load_mha(module.self_attention, f"{prefix}.self_attention")

    def load_group_attn(module, prefix: str):
        load_rms_norm(module.layer_norm, f"{prefix}.layer_norm")
        load_mha(module.self_attention, f"{prefix}.self_attention")

    def load_ffn(module, prefix: str):
        load_rms_norm(module.layer_norm, f"{prefix}.layer_norm")
        load_linear(module.wi, f"{prefix}.mlp.wi")
        load_linear(module.wo, f"{prefix}.mlp.wo")

    def load_residual_block(module, prefix: str):
        load_linear(module.hidden_layer, f"{prefix}.hidden_layer")
        load_linear(module.output_layer, f"{prefix}.output_layer")
        load_linear(module.residual_layer, f"{prefix}.residual_layer")

    # shared embedding
    model.shared.weight = get("shared.weight")

    # input / output patch embeddings
    load_residual_block(model.input_patch_embedding, "input_patch_embedding")
    load_residual_block(model.output_patch_embedding, "output_patch_embedding")

    # encoder blocks: encoder.block.{i}.layer.{0,1,2}
    for i, block in enumerate(model.encoder.block):
        prefix = f"encoder.block.{i}"
        load_time_attn(block.layer[0], f"{prefix}.layer.0")
        load_group_attn(block.layer[1], f"{prefix}.layer.1")
        load_ffn(block.layer[2], f"{prefix}.layer.2")

    # final layer norm
    load_rms_norm(model.encoder.final_layer_norm, "encoder.final_layer_norm")

    return used


def verify_weights_loaded(model: Chronos2MLXModel, weights: dict, used: set[str]) -> None:
    unused = set(weights.keys()) - used
    if unused:
        raise RuntimeError(f"Weight keys not loaded into MLX model:\n" + "\n".join(sorted(unused)))
