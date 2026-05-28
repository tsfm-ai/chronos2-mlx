"""Benchmarking utilities for chronos2-mlx."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx
import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    context_length: int
    prediction_length: int
    batch_rows: int
    dtype: str
    compiled: bool
    mean_ms: float
    min_ms: float
    max_ms: float
    series_per_sec: float
    peak_memory_mb: float = 0.0
    notes: str = ""

    def __str__(self) -> str:
        return (
            f"{self.name} | ctx={self.context_length} pred={self.prediction_length} "
            f"batch={self.batch_rows} {self.dtype}{'(compiled)' if self.compiled else ''} | "
            f"{self.mean_ms:.1f}ms mean | {self.series_per_sec:.1f} series/s"
        )


def time_mlx_call(fn: Callable, *args, warmup: int = 3, repeat: int = 10) -> dict:
    """Time an MLX call with forced evaluation. Returns timing stats in ms."""
    for _ in range(warmup):
        out = fn(*args)
        mx.eval(out)

    times_ms = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    return {
        "mean": float(np.mean(times_ms)),
        "min": float(np.min(times_ms)),
        "max": float(np.max(times_ms)),
        "std": float(np.std(times_ms)),
    }


def benchmark_pipeline(
    pipeline,
    context_lengths: list[int] = [512, 2048],
    prediction_lengths: list[int] = [24, 168],
    batch_sizes: list[int] = [1, 11],
    dtype: str = "float32",
    compiled: bool = False,
    warmup: int = 3,
    repeat: int = 10,
) -> list[BenchmarkResult]:
    results = []
    rng = np.random.default_rng(0)

    for ctx_len in context_lengths:
        for pred_len in prediction_lengths:
            for batch in batch_sizes:
                ctx = mx.array(rng.standard_normal((batch, ctx_len)).astype(np.float32))
                import math
                num_patches = math.ceil(pred_len / pipeline.config.output_patch_size)

                def _forward(c):
                    return pipeline.model(c, num_output_patches=num_patches)

                forward_fn = mx.compile(_forward) if compiled else _forward

                timing = time_mlx_call(forward_fn, ctx, warmup=warmup, repeat=repeat)

                results.append(BenchmarkResult(
                    name="chronos2-mlx",
                    context_length=ctx_len,
                    prediction_length=pred_len,
                    batch_rows=batch,
                    dtype=dtype,
                    compiled=compiled,
                    mean_ms=timing["mean"],
                    min_ms=timing["min"],
                    max_ms=timing["max"],
                    series_per_sec=batch / (timing["mean"] / 1000),
                ))

    return results


def print_benchmark_table(results: list[BenchmarkResult]) -> None:
    print(f"{'ctx':>6} {'pred':>6} {'batch':>6} {'dtype':>8} {'compiled':>9} | {'mean_ms':>9} {'min_ms':>9} {'series/s':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.context_length:>6} {r.prediction_length:>6} {r.batch_rows:>6} "
            f"{r.dtype:>8} {'yes' if r.compiled else 'no':>9} | "
            f"{r.mean_ms:>9.1f} {r.min_ms:>9.1f} {r.series_per_sec:>10.1f}"
        )
