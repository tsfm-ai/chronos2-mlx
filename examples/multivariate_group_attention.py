"""
Multivariate forecast using group attention.

Chronos-2's group attention lets related series share information during
encoding. This example shows how passing a batch of related series produces
different (and typically better) forecasts than encoding each series alone.

Usage:
    python examples/multivariate_group_attention.py
"""
import numpy as np
import mlx.core as mx
from chronos2_mlx import Chronos2MLXPipeline

pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
rng = np.random.default_rng(42)

# ── Simulate two correlated electricity zones ────────────────────────────────
common = np.sin(np.arange(256) / 12) * 5  # shared diurnal pattern
zone_a = (common + rng.standard_normal(256) + 50).astype(np.float32)
zone_b = (common * 1.3 + rng.standard_normal(256) * 0.8 + 45).astype(np.float32)

# Stack as a batch — both series passed together so group attention fires
context = mx.array(np.stack([zone_a, zone_b]))  # [2, 256]

PRED_LEN = 24
quantiles_joint = pipe.predict(context, prediction_length=PRED_LEN)
mx.eval(quantiles_joint)

quantile_levels = np.array(pipe.model.quantiles.astype(mx.float32))
median_idx = int(np.argmin(np.abs(quantile_levels - 0.5)))

q = np.array(quantiles_joint.astype(mx.float32))  # [2, 21, 24]

print("Joint forecast (group attention on):")
print(f"{'Step':>4}  {'Zone A median':>14}  {'Zone B median':>14}")
for i in range(PRED_LEN):
    print(f"{i+1:>4}  {q[0, median_idx, i]:>14.3f}  {q[1, median_idx, i]:>14.3f}")

# ── Compare: encode each series independently ────────────────────────────────
qa = np.array(pipe.predict(context[[0]], prediction_length=PRED_LEN).astype(mx.float32))
qb = np.array(pipe.predict(context[[1]], prediction_length=PRED_LEN).astype(mx.float32))

diff_a = np.abs(q[0, median_idx] - qa[0, median_idx]).mean()
diff_b = np.abs(q[1, median_idx] - qb[0, median_idx]).mean()
print(f"\nMean absolute diff vs independent encoding: Zone A={diff_a:.4f}  Zone B={diff_b:.4f}")
print("(Non-zero means group attention is sharing information between series.)")
