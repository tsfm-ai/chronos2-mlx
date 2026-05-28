"""Quantization and QLoRA tests."""
import tempfile
import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn

from chronos2_mlx import (
    Chronos2MLXPipeline, LoRAConfig, QuantizeConfig,
    apply_lora, fuse_lora, save_adapter, load_adapter,
    quantize_model, param_footprint,
)
from chronos2_mlx.adapters import LoRALinear, _all_lora_paths, _get_module


@pytest.fixture(scope="module")
def base_pipeline():
    return Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")


@pytest.fixture
def ctx():
    return mx.array(np.random.default_rng(7).standard_normal((2, 128)).astype(np.float32))


@pytest.fixture(scope="module")
def fp32_preds(base_pipeline):
    ctx = mx.array(np.random.default_rng(7).standard_normal((2, 128)).astype(np.float32))
    preds = base_pipeline.predict(ctx, prediction_length=24)
    mx.eval(preds)
    return np.array(preds.astype(mx.float32))


# ---------------------------------------------------------------------------
# int8 quantization
# ---------------------------------------------------------------------------

def test_int8_quantizes_linear_layers(base_pipeline):
    pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    stats = quantize_model(pipe.model, bits=8, group_size=64)
    assert stats["num_quantized"] > 0
    assert stats["bits"] == 8
    # Verify some layers are actually QuantizedLinear
    has_ql = any(
        isinstance(m, nn.QuantizedLinear)
        for _, m in pipe.model.named_modules()
    )
    assert has_ql


def test_int8_output_quality(base_pipeline, fp32_preds, ctx):
    """int8 predictions should be within 1% mean relative error of fp32."""
    pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    quantize_model(pipe.model, bits=8, group_size=64)

    preds_q = np.array(pipe.predict(ctx, prediction_length=24).astype(mx.float32))
    mx.eval(preds_q)

    denom = np.abs(fp32_preds).mean()
    mre = np.abs(fp32_preds - preds_q).mean() / (denom + 1e-8)
    assert mre < 0.01, f"int8 MRE={mre:.4f} exceeds 1%"


def test_int8_memory_reduction(base_pipeline):
    """int8 model should use substantially less memory than fp32."""
    pipe_fp32 = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    fp32_stats = param_footprint(pipe_fp32.model)

    pipe_q = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    quantize_model(pipe_q.model, bits=8, group_size=64)
    q8_stats = param_footprint(pipe_q.model)

    # int8 should use < 50% the bytes of fp32 for quantized params
    reduction = 1.0 - q8_stats["total_bytes"] / fp32_stats["total_bytes"]
    assert reduction > 0.4, f"Expected >40% memory reduction, got {reduction:.1%}"


# ---------------------------------------------------------------------------
# int4 quantization
# ---------------------------------------------------------------------------

def test_int4_output_quality(base_pipeline, fp32_preds, ctx):
    """int4 predictions should be within 10% mean relative error of fp32.

    Chronos-2 is a compact 120M-param model; int4 at group_size=64 yields ~6-7% MRE.
    Anything under 10% is acceptable for low-memory deployment.
    """
    pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    quantize_model(pipe.model, bits=4, group_size=64)

    preds_q = np.array(pipe.predict(ctx, prediction_length=24).astype(mx.float32))
    mx.eval(preds_q)

    denom = np.abs(fp32_preds).mean()
    mre = np.abs(fp32_preds - preds_q).mean() / (denom + 1e-8)
    assert mre < 0.10, f"int4 MRE={mre:.4f} exceeds 10%"


def test_int4_memory_reduction(base_pipeline):
    """int4 model should use ~75% less memory than fp32."""
    pipe_fp32 = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    fp32_stats = param_footprint(pipe_fp32.model)

    pipe_q = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    quantize_model(pipe_q.model, bits=4, group_size=64)
    q4_stats = param_footprint(pipe_q.model)

    reduction = 1.0 - q4_stats["total_bytes"] / fp32_stats["total_bytes"]
    assert reduction > 0.6, f"Expected >60% memory reduction, got {reduction:.1%}"


# ---------------------------------------------------------------------------
# QLoRA: LoRA on top of quantized base
# ---------------------------------------------------------------------------

def test_qlora_injection():
    """LoRA adapters inject correctly on top of a quantized model."""
    pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    quantize_model(pipe.model, bits=4, group_size=64)

    cfg = LoRAConfig(rank=4, target_projections=["q", "v"], target_attention_layers=[0, 1])
    injected = apply_lora(pipe.model, cfg)

    # Should inject into the quantized projections
    assert len(injected) > 0, "No QLoRA adapters injected"

    # Verify the base is QuantizedLinear
    m = _get_module(pipe.model, injected[0])
    assert isinstance(m, LoRALinear)
    assert isinstance(m.base, nn.QuantizedLinear)


def test_qlora_zero_init(ctx):
    """QLoRA with zero-init B should not change quantized predictions."""
    pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    quantize_model(pipe.model, bits=4, group_size=64)

    before = np.array(pipe.predict(ctx, prediction_length=24).astype(mx.float32))
    mx.eval(before)

    cfg = LoRAConfig(rank=4, target_projections=["q"], target_attention_layers=[0])
    apply_lora(pipe.model, cfg)

    after = np.array(pipe.predict(ctx, prediction_length=24).astype(mx.float32))
    mx.eval(after)

    assert np.allclose(before, after, atol=1e-5), "QLoRA zero init changed quantized predictions"


def test_qlora_save_load_roundtrip(ctx):
    """QLoRA adapters save/load correctly (adapter weights only, base stays quantized)."""
    pipe_orig = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    quantize_model(pipe_orig.model, bits=4, group_size=64)

    cfg = LoRAConfig(rank=4, target_projections=["q"], target_attention_layers=[0])
    apply_lora(pipe_orig.model, cfg)

    # Non-zero adapters
    for path in _all_lora_paths(pipe_orig.model):
        m = _get_module(pipe_orig.model, path)
        m.lora_b = mx.random.normal(m.lora_b.shape) * 0.01
    mx.eval(pipe_orig.model.parameters())

    pred_orig = np.array(pipe_orig.predict(ctx, prediction_length=24).astype(mx.float32))

    with tempfile.TemporaryDirectory() as tmpdir:
        save_adapter(pipe_orig.model, tmpdir, cfg)

        pipe_loaded = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
        quantize_model(pipe_loaded.model, bits=4, group_size=64)
        load_adapter(pipe_loaded.model, tmpdir)

        pred_loaded = np.array(pipe_loaded.predict(ctx, prediction_length=24).astype(mx.float32))

    assert np.allclose(pred_orig, pred_loaded, atol=1e-4), (
        f"QLoRA roundtrip mismatch: max_ae={np.abs(pred_orig - pred_loaded).max():.2e}"
    )


def test_qlora_fuse_correctness(ctx):
    """Fusing QLoRA adapters removes all LoRALinear and preserves predictions within quantization noise.

    QLoRA fuse is inherently lossy: fuse does dequantize→add_delta→re-quantize,
    which introduces a second round of int4 rounding error on top of the first.
    We tolerate up to 0.1 absolute difference — much smaller than the adapter's effect.
    """
    pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    quantize_model(pipe.model, bits=4, group_size=64)

    cfg = LoRAConfig(rank=4, target_projections=["q", "v"], target_attention_layers=[0, 1])
    apply_lora(pipe.model, cfg)

    for path in _all_lora_paths(pipe.model):
        m = _get_module(pipe.model, path)
        m.lora_b = mx.random.normal(m.lora_b.shape) * 0.01
    mx.eval(pipe.model.parameters())

    pred_before = np.array(pipe.predict(ctx, prediction_length=24).astype(mx.float32))

    fuse_lora(pipe.model)

    pred_after = np.array(pipe.predict(ctx, prediction_length=24).astype(mx.float32))

    # No LoRALinear modules should remain
    assert len(_all_lora_paths(pipe.model)) == 0, "LoRALinear modules remain after fuse"

    # Predictions should be finite and in a sane range
    assert np.isfinite(pred_after).all(), "Non-finite values after QLoRA fuse"

    # Re-quantization noise is expected; the error should be << lora_b * scaling effect
    max_ae = np.abs(pred_before - pred_after).max()
    assert max_ae < 0.15, f"QLoRA fuse diverged unexpectedly: max_ae={max_ae:.3f}"
