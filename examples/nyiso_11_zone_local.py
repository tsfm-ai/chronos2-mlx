"""
NYISO 11-zone joint forecast with chronos2-mlx.

Shows how to use chronos2-mlx as a local Apple Silicon inference engine for
multi-zone energy load forecasting — feeding all 11 NYISO load zones in a
single batch so group attention can share cross-zone information.

This example generates synthetic data shaped like real NYISO hourly load.
Swap in real data by loading your DataFrame and converting to numpy arrays.

Usage:
    python examples/nyiso_11_zone_local.py
"""
import numpy as np
import mlx.core as mx
from chronos2_mlx import Chronos2MLXPipeline, QuantizeConfig, quantize_model

ZONES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
CONTEXT_HOURS = 168  # one week of hourly data
PRED_HOURS = 24

print("Loading model (int8 for lower memory)...")
pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2", dtype=mx.bfloat16)
quantize_model(pipe.model, bits=8, group_size=64)

rng = np.random.default_rng(0)

# ── Simulate realistic zone loads ────────────────────────────────────────────
t = np.arange(CONTEXT_HOURS, dtype=np.float32)
diurnal = np.sin(2 * np.pi * t / 24)         # 24h cycle
weekly = np.sin(2 * np.pi * t / 168) * 0.3   # 7-day cycle

zones_data = []
base_loads = [3200, 1800, 2100, 1600, 1200, 4100, 5500, 1400, 900, 11000, 3000]
for base in base_loads:
    noise = rng.standard_normal(CONTEXT_HOURS).astype(np.float32) * 0.03 * base
    zone = (base * (1 + 0.12 * diurnal + weekly)).astype(np.float32) + noise
    zones_data.append(zone)

context = mx.array(np.stack(zones_data))  # [11, 168]

print(f"Running joint forecast: 11 zones × {PRED_HOURS}h horizon...")
quantiles = pipe.predict(context, prediction_length=PRED_HOURS)
mx.eval(quantiles)

quantile_levels = np.array(pipe.model.quantiles.astype(mx.float32))
median_idx = int(np.argmin(np.abs(quantile_levels - 0.5)))
lo_idx = int(np.argmin(np.abs(quantile_levels - 0.1)))
hi_idx = int(np.argmin(np.abs(quantile_levels - 0.9)))

q = np.array(quantiles.astype(mx.float32))  # [11, 21, 24]

print(f"\n{'Zone':<6}  {'24h mean MW':>12}  {'P10 mean':>10}  {'P90 mean':>10}")
for i, zone in enumerate(ZONES):
    median = q[i, median_idx].mean()
    lo = q[i, lo_idx].mean()
    hi = q[i, hi_idx].mean()
    print(f"  {zone:<4}  {median:>12.0f}  {lo:>10.0f}  {hi:>10.0f}")

nyiso_total_median = q[:, median_idx, :].sum(axis=0)
print(f"\nNYISO total: peak={nyiso_total_median.max():.0f} MW  mean={nyiso_total_median.mean():.0f} MW")
