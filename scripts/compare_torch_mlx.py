"""
Parity comparison: PyTorch Chronos-2 vs MLX port.
Run with: python scripts/compare_torch_mlx.py
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import torch

from chronos import Chronos2Pipeline as TorchPipeline
from chronos2_mlx import Chronos2MLXPipeline
import mlx.core as mx


def rel_err(a, b):
    denom = np.abs(b).mean() + 1e-8
    return np.abs(a - b).mean() / denom


def check(name, mlx_arr, torch_arr, atol=1e-3, rtol=1e-3):
    m = np.array(mlx_arr)
    t = torch_arr.float().numpy()
    ae = np.abs(m - t)
    re = rel_err(m, t)
    ok = np.allclose(m, t, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:50s}  max_ae={ae.max():.2e}  mean_ae={ae.mean():.2e}  rel={re:.2e}")
    return ok


def main():
    print("=" * 70)
    print("Loading PyTorch Chronos-2...")
    torch_pipe = TorchPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        attn_implementation="eager",
    )
    torch_model = torch_pipe.model.eval()

    print("Loading MLX Chronos-2...")
    mlx_pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
    mlx_model = mlx_pipe.model

    print("=" * 70)
    print("Running parity tests...")
    print()

    rng = np.random.default_rng(42)
    BATCH = 3
    CTX = 128

    # Test case 1: basic univariate (no NaN)
    ctx_np = rng.standard_normal((BATCH, CTX)).astype(np.float32)
    ctx_torch = torch.tensor(ctx_np)
    ctx_mlx = mx.array(ctx_np)
    PRED = 24

    results = []

    # --- Layer-level: instance norm ---
    print("--- InstanceNorm ---")
    with torch.no_grad():
        normed_torch, (loc_t, scale_t) = torch_model.instance_norm(ctx_torch)

    normed_mlx, (loc_m, scale_m) = mlx_model.instance_norm(ctx_mlx)
    mx.eval(normed_mlx, loc_m, scale_m)

    results.append(check("instance_norm / loc",    loc_m,    loc_t,    atol=1e-5))
    results.append(check("instance_norm / scale",  scale_m,  scale_t,  atol=1e-5))
    results.append(check("instance_norm / output", normed_mlx, normed_torch, atol=1e-5))

    # --- Layer-level: patching ---
    print("\n--- Patching ---")
    from chronos2_mlx.preprocessing import patch_sequence

    ctx_masked = ctx_np.copy()
    # keep clean for patching test
    patched_torch = torch_model.patch(normed_torch)
    patched_mlx = patch_sequence(normed_mlx, 16, 16)
    mx.eval(patched_mlx)

    results.append(check("patching", patched_mlx, patched_torch, atol=1e-5))

    # --- Layer-level: full _prepare_patched_context ---
    print("\n--- _prepare_patched_context ---")
    with torch.no_grad():
        ptc_torch, attn_mask_torch, (loc_t, scale_t) = torch_model._prepare_patched_context(ctx_torch)
    ptc_mlx, attn_mask_mlx, (loc_m, scale_m) = mlx_model._prepare_patched_context(ctx_mlx)
    mx.eval(ptc_mlx, attn_mask_mlx)

    results.append(check("patched_context", ptc_mlx, ptc_torch, atol=1e-5))
    results.append(check("attn_mask",       attn_mask_mlx, attn_mask_torch.float(), atol=1e-5))

    # --- Layer-level: input_patch_embedding ---
    print("\n--- input_patch_embedding ---")
    with torch.no_grad():
        emb_torch = torch_model.input_patch_embedding(ptc_torch)
    emb_mlx = mlx_model.input_patch_embedding(ptc_mlx)
    mx.eval(emb_mlx)

    results.append(check("input_patch_embedding", emb_mlx, emb_torch, atol=1e-4))

    # --- Layer-level: shared embedding (REG token) ---
    print("\n--- REG token embedding ---")
    reg_ids_torch = torch.full((BATCH, 1), 1, dtype=torch.long)
    with torch.no_grad():
        reg_emb_torch = torch_model.shared(reg_ids_torch)
    reg_ids_mlx = mx.full((BATCH, 1), 1, dtype=mx.int32)
    reg_emb_mlx = mlx_model.shared(reg_ids_mlx)
    mx.eval(reg_emb_mlx)

    results.append(check("reg_token_embedding", reg_emb_mlx, reg_emb_torch, atol=1e-5))

    # --- Full encode() pass ---
    print("\n--- encode() hidden states ---")
    num_output_patches = 2  # 32 timesteps
    with torch.no_grad():
        enc_out_torch, (loc_t, scale_t), _, num_ctx_patches = torch_model.encode(
            context=ctx_torch, num_output_patches=num_output_patches
        )
        hs_torch = enc_out_torch[0]  # [batch, seq, d_model]

    hs_mlx, (loc_m, scale_m), _, _ = mlx_model.encode(
        context=ctx_mlx, num_output_patches=num_output_patches
    )
    mx.eval(hs_mlx, loc_m, scale_m)

    results.append(check("encoder hidden states (loc)",    loc_m,  loc_t,  atol=1e-5))
    results.append(check("encoder hidden states (scale)",  scale_m, scale_t, atol=1e-5))
    results.append(check("encoder hidden states",          hs_mlx, hs_torch, atol=5e-3))

    # --- Full forward: quantile predictions ---
    print("\n--- forward() quantile predictions ---")
    with torch.no_grad():
        out_torch = torch_model(context=ctx_torch, num_output_patches=num_output_patches)
        qp_torch = out_torch.quantile_preds  # [batch, num_quantiles, horizon]

    qp_mlx = mlx_model(context=ctx_mlx, num_output_patches=num_output_patches)
    mx.eval(qp_mlx)

    results.append(check("quantile_preds (all)",   qp_mlx, qp_torch, atol=1e-2))
    results.append(check("quantile_preds (p10)",   qp_mlx[:, 2, :], qp_torch[:, 2, :], atol=1e-2))
    results.append(check("quantile_preds (p50)",   qp_mlx[:, 10, :], qp_torch[:, 10, :], atol=1e-2))
    results.append(check("quantile_preds (p90)",   qp_mlx[:, 18, :], qp_torch[:, 18, :], atol=1e-2))

    # --- Missing values test ---
    print("\n--- Missing values ---")
    ctx_nan = ctx_np.copy()
    ctx_nan[0, 20:40] = np.nan
    ctx_nan[1, :10] = np.nan

    with torch.no_grad():
        out_nan_torch = torch_model(
            context=torch.tensor(ctx_nan), num_output_patches=num_output_patches
        )
        qp_nan_torch = out_nan_torch.quantile_preds

    qp_nan_mlx = mlx_model(context=mx.array(ctx_nan), num_output_patches=num_output_patches)
    mx.eval(qp_nan_mlx)

    results.append(check("missing_values p50", qp_nan_mlx[:, 10, :], qp_nan_torch[:, 10, :], atol=2e-2))

    print()
    print("=" * 70)
    n_pass = sum(results)
    n_total = len(results)
    print(f"SUMMARY: {n_pass}/{n_total} tests passed")
    if n_pass < n_total:
        print("SOME TESTS FAILED — parity not achieved yet")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED — MLX parity confirmed")


if __name__ == "__main__":
    main()
