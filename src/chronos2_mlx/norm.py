"""T5-style RMS layer norm (variance only, no mean subtraction, no bias)."""
import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """Matches Chronos2LayerNorm exactly: weight only, variance computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x32 = x.astype(mx.float32)
        variance = mx.mean(mx.square(x32), axis=-1, keepdims=True)
        x32 = x32 * mx.rsqrt(variance + self.eps)
        if orig_dtype in (mx.float16, mx.bfloat16):
            x32 = x32.astype(orig_dtype)
        return x32 * self.weight
