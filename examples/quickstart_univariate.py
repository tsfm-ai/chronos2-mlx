"""
Quickstart: univariate point forecast with chronos2-mlx.

Loads amazon/chronos-2 from HuggingFace, forecasts a synthetic time series,
and prints the median and 90% prediction interval.

Usage:
    python examples/quickstart_univariate.py
"""
import numpy as np
import mlx.core as mx
from chronos2_mlx import Chronos2MLXPipeline

# ── Load model ──────────────────────────────────────────────────────────────
pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")

# ── Build a context window ───────────────────────────────────────────────────
rng = np.random.default_rng(0)
# Noisy sine wave with trend — shape [1, context_length]
t = np.arange(512)
values = (np.sin(t / 20) * 10 + t * 0.05 + rng.standard_normal(512)).astype(np.float32)
context = mx.array(values[np.newaxis, :])  # [1, 512]

# ── Predict ──────────────────────────────────────────────────────────────────
PRED_LEN = 24
quantiles = pipe.predict(context, prediction_length=PRED_LEN)  # [1, num_quantiles, horizon]
mx.eval(quantiles)

q = np.array(quantiles[0].astype(mx.float32))  # [21, 24]
quantile_levels = np.array(pipe.model.quantiles.astype(mx.float32))

median_idx = int(np.argmin(np.abs(quantile_levels - 0.5)))
lo_idx = int(np.argmin(np.abs(quantile_levels - 0.05)))
hi_idx = int(np.argmin(np.abs(quantile_levels - 0.95)))

median = q[median_idx]
lo = q[lo_idx]
hi = q[hi_idx]

print(f"Forecast horizon: {PRED_LEN} steps")
print(f"{'Step':>4}  {'Median':>10}  {'P05':>10}  {'P95':>10}")
for i in range(PRED_LEN):
    print(f"{i+1:>4}  {median[i]:>10.3f}  {lo[i]:>10.3f}  {hi[i]:>10.3f}")
