"""Layer-by-layer and full-model parity tests: MLX vs PyTorch Chronos-2."""
import numpy as np
import pytest
import torch
import mlx.core as mx

from .conftest import check_close
from chronos2_mlx.preprocessing import patch_sequence

pytestmark = pytest.mark.parity


RNG = np.random.default_rng(42)
BATCH = 3
CTX = 128
NUM_OUTPUT_PATCHES = 2


@pytest.fixture(scope="module")
def ctx_np():
    return RNG.standard_normal((BATCH, CTX)).astype(np.float32)


@pytest.fixture(scope="module")
def ctx_torch(ctx_np):
    return torch.tensor(ctx_np)


@pytest.fixture(scope="module")
def ctx_mlx(ctx_np):
    return mx.array(ctx_np)


# --- InstanceNorm ---

def test_instance_norm_loc(torch_model, mlx_model, ctx_torch, ctx_mlx):
    with torch.no_grad():
        _, (loc_t, _) = torch_model.instance_norm(ctx_torch)
    _, (loc_m, _) = mlx_model.instance_norm(ctx_mlx)
    mx.eval(loc_m)
    check_close(loc_m, loc_t, atol=1e-5, name="instance_norm/loc")


def test_instance_norm_scale(torch_model, mlx_model, ctx_torch, ctx_mlx):
    with torch.no_grad():
        _, (_, scale_t) = torch_model.instance_norm(ctx_torch)
    _, (_, scale_m) = mlx_model.instance_norm(ctx_mlx)
    mx.eval(scale_m)
    check_close(scale_m, scale_t, atol=1e-5, name="instance_norm/scale")


def test_instance_norm_output(torch_model, mlx_model, ctx_torch, ctx_mlx):
    with torch.no_grad():
        normed_t, _ = torch_model.instance_norm(ctx_torch)
    normed_m, _ = mlx_model.instance_norm(ctx_mlx)
    mx.eval(normed_m)
    check_close(normed_m, normed_t, atol=1e-5, name="instance_norm/output")


# --- Patching ---

def test_patching(torch_model, mlx_model, ctx_torch, ctx_mlx):
    with torch.no_grad():
        normed_t, _ = torch_model.instance_norm(ctx_torch)
        patched_t = torch_model.patch(normed_t)
    normed_m, _ = mlx_model.instance_norm(ctx_mlx)
    patched_m = patch_sequence(normed_m, 16, 16)
    mx.eval(patched_m)
    check_close(patched_m, patched_t, atol=1e-5, name="patching")


# --- _prepare_patched_context ---

def test_patched_context(torch_model, mlx_model, ctx_torch, ctx_mlx):
    with torch.no_grad():
        ptc_t, _, _ = torch_model._prepare_patched_context(ctx_torch)
    ptc_m, _, _ = mlx_model._prepare_patched_context(ctx_mlx)
    mx.eval(ptc_m)
    check_close(ptc_m, ptc_t, atol=1e-5, name="patched_context")


def test_attention_mask(torch_model, mlx_model, ctx_torch, ctx_mlx):
    with torch.no_grad():
        _, attn_t, _ = torch_model._prepare_patched_context(ctx_torch)
    _, attn_m, _ = mlx_model._prepare_patched_context(ctx_mlx)
    mx.eval(attn_m)
    check_close(attn_m, attn_t.float(), atol=1e-5, name="attention_mask")


# --- input_patch_embedding ---

def test_input_patch_embedding(torch_model, mlx_model, ctx_torch, ctx_mlx):
    with torch.no_grad():
        ptc_t, _, _ = torch_model._prepare_patched_context(ctx_torch)
        emb_t = torch_model.input_patch_embedding(ptc_t)
    ptc_m, _, _ = mlx_model._prepare_patched_context(ctx_mlx)
    emb_m = mlx_model.input_patch_embedding(ptc_m)
    mx.eval(emb_m)
    check_close(emb_m, emb_t, atol=1e-4, name="input_patch_embedding")


# --- Encoder hidden states ---

def test_encoder_hidden_states(torch_model, mlx_model, ctx_torch, ctx_mlx):
    with torch.no_grad():
        enc_t, _, _, _ = torch_model.encode(context=ctx_torch, num_output_patches=NUM_OUTPUT_PATCHES)
        hs_t = enc_t[0]
    hs_m, _, _, _ = mlx_model.encode(context=ctx_mlx, num_output_patches=NUM_OUTPUT_PATCHES)
    mx.eval(hs_m)
    check_close(hs_m, hs_t, atol=5e-3, name="encoder_hidden_states")


# --- Full model forward ---

def test_quantile_preds_all(torch_model, mlx_model, ctx_torch, ctx_mlx):
    with torch.no_grad():
        qp_t = torch_model(context=ctx_torch, num_output_patches=NUM_OUTPUT_PATCHES).quantile_preds
    qp_m = mlx_model(context=ctx_mlx, num_output_patches=NUM_OUTPUT_PATCHES)
    mx.eval(qp_m)
    check_close(qp_m, qp_t, atol=1e-2, name="quantile_preds")


def test_quantile_p50(torch_model, mlx_model, ctx_torch, ctx_mlx):
    with torch.no_grad():
        qp_t = torch_model(context=ctx_torch, num_output_patches=NUM_OUTPUT_PATCHES).quantile_preds[:, 10, :]
    qp_m = mlx_model(context=ctx_mlx, num_output_patches=NUM_OUTPUT_PATCHES)[:, 10, :]
    mx.eval(qp_m)
    check_close(qp_m, qp_t, atol=1e-3, name="p50")


# --- Missing values ---

def test_missing_values(torch_model, mlx_model):
    ctx_nan = RNG.standard_normal((2, CTX)).astype(np.float32)
    ctx_nan[0, 20:40] = np.nan
    ctx_nan[1, :10] = np.nan

    with torch.no_grad():
        qp_t = torch_model(
            context=torch.tensor(ctx_nan), num_output_patches=NUM_OUTPUT_PATCHES
        ).quantile_preds[:, 10, :]

    qp_m = mlx_model(context=mx.array(ctx_nan), num_output_patches=NUM_OUTPUT_PATCHES)[:, 10, :]
    mx.eval(qp_m)
    check_close(qp_m, qp_t, atol=2e-2, name="missing_values_p50")


# --- Group attention (cross-series) ---

def test_group_attention_independent(torch_model, mlx_model, ctx_torch, ctx_mlx):
    """With distinct group_ids, each series is independent — same as default."""
    group_ids_t = torch.arange(BATCH, dtype=torch.long)
    group_ids_m = mx.arange(BATCH, dtype=mx.int32)

    with torch.no_grad():
        qp_t = torch_model(
            context=ctx_torch, group_ids=group_ids_t, num_output_patches=NUM_OUTPUT_PATCHES
        ).quantile_preds

    qp_m = mlx_model(context=ctx_mlx, group_ids=group_ids_m, num_output_patches=NUM_OUTPUT_PATCHES)
    mx.eval(qp_m)
    check_close(qp_m, qp_t, atol=1e-2, name="group_independent")


def test_group_attention_shared(torch_model, mlx_model, ctx_torch, ctx_mlx):
    """With same group_id, all series share information — cross-learning mode."""
    group_ids_t = torch.zeros(BATCH, dtype=torch.long)
    group_ids_m = mx.zeros((BATCH,), dtype=mx.int32)

    with torch.no_grad():
        qp_t = torch_model(
            context=ctx_torch, group_ids=group_ids_t, num_output_patches=NUM_OUTPUT_PATCHES
        ).quantile_preds

    qp_m = mlx_model(context=ctx_mlx, group_ids=group_ids_m, num_output_patches=NUM_OUTPUT_PATCHES)
    mx.eval(qp_m)
    check_close(qp_m, qp_t, atol=1e-2, name="group_shared")


# --- Long context ---

def test_long_context(torch_model, mlx_model):
    ctx_long = RNG.standard_normal((2, 2048)).astype(np.float32)
    with torch.no_grad():
        qp_t = torch_model(
            context=torch.tensor(ctx_long), num_output_patches=4
        ).quantile_preds[:, 10, :]

    qp_m = mlx_model(context=mx.array(ctx_long), num_output_patches=4)[:, 10, :]
    mx.eval(qp_m)
    check_close(qp_m, qp_t, atol=1e-2, name="long_context_p50")


# --- Future covariates ---

def test_future_covariates(torch_model, mlx_model, ctx_torch, ctx_mlx):
    future_len = NUM_OUTPUT_PATCHES * 16
    fut_np = RNG.standard_normal((BATCH, future_len)).astype(np.float32)
    fut_t = torch.tensor(fut_np)
    fut_m = mx.array(fut_np)

    with torch.no_grad():
        qp_t = torch_model(
            context=ctx_torch,
            future_covariates=fut_t,
            num_output_patches=NUM_OUTPUT_PATCHES,
        ).quantile_preds[:, 10, :]

    qp_m = mlx_model(
        context=ctx_mlx,
        future_covariates=fut_m,
        num_output_patches=NUM_OUTPUT_PATCHES,
    )[:, 10, :]
    mx.eval(qp_m)
    check_close(qp_m, qp_t, atol=2e-2, name="future_covariates_p50")
