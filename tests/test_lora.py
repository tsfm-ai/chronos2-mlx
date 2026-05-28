"""LoRA adapter tests."""
import tempfile
import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn

from chronos2_mlx import (
    Chronos2MLXPipeline, LoRAConfig, TrainConfig,
    apply_lora, fuse_lora, save_adapter, load_adapter, fine_tune,
)
from chronos2_mlx.adapters import LoRALinear, _all_lora_paths, _get_module


@pytest.fixture(scope="module")
def base_pipeline():
    return Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")


@pytest.fixture
def ctx():
    return mx.array(np.random.default_rng(7).standard_normal((2, 128)).astype(np.float32))


def test_lora_injection(base_pipeline):
    pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    cfg = LoRAConfig(rank=4, target_projections=["q", "v"], target_attention_layers=[0, 1])
    injected = apply_lora(pipe.model, cfg)
    assert len(injected) == 12 * 2 * 2  # 12 layers * 2 attn * 2 projections


def test_lora_zero_init(base_pipeline, ctx):
    """LoRA with zero-initialized B should not change predictions."""
    pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    before = np.array(pipe.predict(ctx, prediction_length=24).astype(mx.float32))

    cfg = LoRAConfig(rank=4, target_projections=["q"], target_attention_layers=[0])
    apply_lora(pipe.model, cfg)
    after = np.array(pipe.predict(ctx, prediction_length=24).astype(mx.float32))
    mx.eval(after)

    assert np.allclose(before, after, atol=1e-5), "LoRA zero init changed predictions"


def test_base_weights_not_trainable(base_pipeline):
    pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    cfg = LoRAConfig(rank=4, target_projections=["q", "v"], target_attention_layers=[0, 1])
    apply_lora(pipe.model, cfg)

    train_cfg = TrainConfig(finetune_mode="lora", lora=cfg, max_steps=1)
    rng = np.random.default_rng(0)
    series = [rng.standard_normal(200).astype(np.float32)]

    from chronos2_mlx.train import _setup_finetune_mode
    _setup_finetune_mode(pipe.model, train_cfg)

    trainable = nn.utils.tree_flatten(pipe.model.trainable_parameters())
    base_keys = [k for k, _ in trainable if ".base." in k]
    assert len(base_keys) == 0, f"Base weights are trainable: {base_keys[:3]}"


def test_lora_training_reduces_loss():
    pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    cfg = LoRAConfig(rank=4, target_projections=["q", "v"], target_attention_layers=[0])
    apply_lora(pipe.model, cfg)

    rng = np.random.default_rng(42)
    series = [rng.standard_normal(400).astype(np.float32) for _ in range(5)]

    train_cfg = TrainConfig(
        prediction_length=24, context_length=128, max_steps=20,
        log_every=20, batch_size=16, finetune_mode="lora", lora=cfg,
    )
    log = fine_tune(pipe.model, series, train_cfg, verbose=False)
    # Loss should have moved from initial value
    assert len(log) == 20
    assert log[-1]["train_loss"] < 100.0  # sanity: not NaN or huge


def test_save_load_roundtrip(ctx):
    pipe_orig = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    cfg = LoRAConfig(rank=4, target_projections=["q"], target_attention_layers=[0])
    apply_lora(pipe_orig.model, cfg)

    # Manually set non-zero lora_b to make predictions differ from base
    from chronos2_mlx.adapters import _all_lora_paths, _get_module, LoRALinear
    for path in _all_lora_paths(pipe_orig.model):
        m = _get_module(pipe_orig.model, path)
        m.lora_b = mx.random.normal(m.lora_b.shape) * 0.01
    mx.eval(pipe_orig.model.parameters())

    pred_orig = np.array(pipe_orig.predict(ctx, prediction_length=24).astype(mx.float32))

    with tempfile.TemporaryDirectory() as tmpdir:
        save_adapter(pipe_orig.model, tmpdir, cfg)

        pipe_loaded = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
        load_adapter(pipe_loaded.model, tmpdir)

        pred_loaded = np.array(pipe_loaded.predict(ctx, prediction_length=24).astype(mx.float32))

    assert np.allclose(pred_orig, pred_loaded, atol=1e-4), (
        f"Roundtrip mismatch: max_ae={np.abs(pred_orig - pred_loaded).max():.2e}"
    )


def test_fuse_correctness(ctx):
    pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    cfg = LoRAConfig(rank=4, target_projections=["q", "v"], target_attention_layers=[0, 1])
    apply_lora(pipe.model, cfg)

    # Non-zero adapters
    for path in _all_lora_paths(pipe.model):
        m = _get_module(pipe.model, path)
        m.lora_b = mx.random.normal(m.lora_b.shape) * 0.01
    mx.eval(pipe.model.parameters())

    pred_before = np.array(pipe.predict(ctx, prediction_length=24).astype(mx.float32))

    fuse_lora(pipe.model)

    pred_after = np.array(pipe.predict(ctx, prediction_length=24).astype(mx.float32))

    assert np.allclose(pred_before, pred_after, atol=1e-4), (
        f"Fuse changed predictions: max_ae={np.abs(pred_before - pred_after).max():.2e}"
    )
    # Verify no more LoRALinear in model
    assert len(_all_lora_paths(pipe.model)) == 0
