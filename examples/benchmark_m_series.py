"""
Apple Silicon performance benchmark for chronos2-mlx.

Sweeps over context lengths and batch sizes to show throughput on M-series chips.
Compares fp32 vs bf16 vs int8.

Usage:
    python examples/benchmark_m_series.py
"""
import time
import numpy as np
import mlx.core as mx
from chronos2_mlx import Chronos2MLXPipeline, QuantizeConfig, quantize_model, param_footprint

PRED_LEN = 24
WARMUP = 2
REPEATS = 5

CONFIGS = [
    ("fp32",  "float32", None),
    ("bf16",  "bfloat16", None),
    ("int8",  "float32", QuantizeConfig(bits=8, group_size=64)),
    ("int4",  "float32", QuantizeConfig(bits=4, group_size=64)),
]


def run_benchmark(pipe, context_len: int, batch: int, repeats: int) -> float:
    """Return median latency in ms."""
    ctx = mx.array(np.random.default_rng(0).standard_normal((batch, context_len)).astype(np.float32))
    times = []
    for _ in range(repeats + WARMUP):
        t0 = time.perf_counter()
        out = pipe.predict(ctx, prediction_length=PRED_LEN)
        mx.eval(out)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    return float(np.median(times[WARMUP:]))


print(f"{'Config':<8} {'ctx':>6} {'batch':>5} {'ms':>8} {'series/s':>10} {'size_MB':>10}")
print("-" * 55)

for label, dtype, qcfg in CONFIGS:
    for context_len in [128, 512]:
        for batch in [1, 8]:
            pipe = Chronos2MLXPipeline.from_pretrained(
                "amazon/chronos-2",
                dtype=getattr(mx, dtype),
            )
            if qcfg is not None:
                quantize_model(pipe.model, qcfg)

            ms = run_benchmark(pipe, context_len, batch, REPEATS)
            sps = batch / (ms / 1000)
            fp = param_footprint(pipe.model)
            size_mb = fp["total_bytes"] / 1e6

            print(f"{label:<8} {context_len:>6} {batch:>5} {ms:>8.1f} {sps:>10.1f} {size_mb:>10.1f}")
