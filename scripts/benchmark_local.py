"""
Apple Silicon benchmark: MLX fp32 vs bf16 vs compiled vs PyTorch CPU.
Run: python scripts/benchmark_local.py
"""
import sys
import platform
sys.path.insert(0, "src")

import numpy as np
import torch
import time

import mlx.core as mx
from chronos2_mlx import Chronos2MLXPipeline
from chronos2_mlx.benchmark import benchmark_pipeline, print_benchmark_table, time_mlx_call


def benchmark_pytorch(model, ctx_len: int, pred_len: int, batch: int, warmup=3, repeat=10):
    import math
    ctx = torch.randn(batch, ctx_len)
    num_patches = math.ceil(pred_len / 16)

    @torch.no_grad()
    def forward():
        return model(context=ctx, num_output_patches=num_patches)

    for _ in range(warmup):
        forward()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        forward()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {"mean": np.mean(times), "min": np.min(times)}


def main():
    print(f"Platform: {platform.machine()} | Python: {platform.python_version()}")
    print(f"MLX device: {mx.default_device()}")
    print()

    CTX_LENS = [512, 2048]
    PRED_LENS = [24, 168]
    BATCHES = [1, 11]

    # --- MLX fp32 ---
    print("Loading MLX fp32...")
    pipe_fp32 = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2", dtype="float32")

    print("\n=== MLX fp32 ===")
    results_fp32 = benchmark_pipeline(
        pipe_fp32, context_lengths=CTX_LENS, prediction_lengths=PRED_LENS,
        batch_sizes=BATCHES, dtype="float32", compiled=False,
    )
    print_benchmark_table(results_fp32)

    # --- MLX fp32 compiled ---
    print("\n=== MLX fp32 (compiled) ===")
    results_compiled = benchmark_pipeline(
        pipe_fp32, context_lengths=CTX_LENS, prediction_lengths=PRED_LENS,
        batch_sizes=BATCHES, dtype="float32", compiled=True,
    )
    print_benchmark_table(results_compiled)

    # --- MLX bf16 ---
    print("\nLoading MLX bf16...")
    pipe_bf16 = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2", dtype="bfloat16")
    print("\n=== MLX bf16 ===")
    results_bf16 = benchmark_pipeline(
        pipe_bf16, context_lengths=CTX_LENS, prediction_lengths=PRED_LENS,
        batch_sizes=BATCHES, dtype="bfloat16", compiled=False,
    )
    print_benchmark_table(results_bf16)

    # --- PyTorch CPU ---
    print("\nLoading PyTorch CPU...")
    from chronos import Chronos2Pipeline
    torch_pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    torch_model = torch_pipe.model.eval()

    print("\n=== PyTorch CPU ===")
    print(f"{'ctx':>6} {'pred':>6} {'batch':>6} {'dtype':>8} {'compiled':>9} | {'mean_ms':>9} {'min_ms':>9} {'series/s':>10}")
    print("-" * 80)
    for ctx_len in CTX_LENS:
        for pred_len in PRED_LENS:
            for batch in BATCHES:
                timing = benchmark_pytorch(torch_model, ctx_len, pred_len, batch)
                series_per_s = batch / (timing["mean"] / 1000)
                print(
                    f"{ctx_len:>6} {pred_len:>6} {batch:>6} {'float32':>8} {'no':>9} | "
                    f"{timing['mean']:>9.1f} {timing['min']:>9.1f} {series_per_s:>10.1f}"
                )

    # --- Speedup summary ---
    print("\n=== Speedup: MLX fp32 vs PyTorch CPU ===")
    for r in results_fp32:
        timing_pt = benchmark_pytorch(torch_model, r.context_length, r.prediction_length, r.batch_rows, warmup=1, repeat=5)
        speedup = timing_pt["mean"] / r.mean_ms
        print(f"ctx={r.context_length} pred={r.prediction_length} batch={r.batch_rows}: {speedup:.2f}x speedup")


if __name__ == "__main__":
    main()
