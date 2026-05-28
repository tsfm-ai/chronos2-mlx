"""Multi-head attention for Chronos-2."""
import mlx.core as mx
import mlx.nn as nn

from .config import Chronos2MLXConfig
from .norm import RMSNorm
from .rope import apply_rope


class MultiHeadAttention(nn.Module):
    """Unscaled dot-product attention (Chronos-2 uses scale=1.0, not 1/sqrt(d_kv))."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        inner = num_heads * d_kv
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)

    def _shape(self, x: mx.array, seq: int) -> mx.array:
        # [batch, seq, inner] -> [batch, heads, seq, d_kv]
        b = x.shape[0]
        x = x.reshape(b, seq, self.num_heads, self.d_kv)
        return mx.transpose(x, (0, 2, 1, 3))

    def _unshape(self, x: mx.array, seq: int) -> mx.array:
        # [batch, heads, seq, d_kv] -> [batch, seq, inner]
        b = x.shape[0]
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(b, seq, self.num_heads * self.d_kv)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        seq = hidden_states.shape[1]

        q = self._shape(self.q(hidden_states), seq)
        k = self._shape(self.k(hidden_states), seq)
        v = self._shape(self.v(hidden_states), seq)

        if self.use_rope:
            assert position_ids is not None
            q, k = apply_rope(q, k, position_ids, theta=self.rope_theta)

        # No scaling — Chronos-2 uses scale=1.0 explicitly
        scores = q @ mx.swapaxes(k, -1, -2)  # [batch, heads, seq, seq]

        if mask is not None:
            scores = scores + mask

        # Softmax in float32 for stability
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(hidden_states.dtype)
        out = weights @ v
        return self.o(self._unshape(out, seq))


class TimeSelfAttention(nn.Module):
    """Time attention with RoPE — attends across patch-time axis per row."""

    def __init__(self, config: Chronos2MLXConfig):
        super().__init__()
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.self_attention = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_kv=config.d_kv,
            use_rope=True,
            rope_theta=config.rope_theta,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        normed = self.layer_norm(hidden_states)
        attn_out = self.self_attention(normed, mask=attention_mask, position_ids=position_ids)
        return hidden_states + attn_out


class GroupSelfAttention(nn.Module):
    """Group attention — attends across rows (same group) at each patch position."""

    def __init__(self, config: Chronos2MLXConfig):
        super().__init__()
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.self_attention = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_kv=config.d_kv,
            use_rope=False,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        # hidden_states: [batch, time, d]
        # Transpose so attention operates along batch (group) dim at each time step
        hs = mx.transpose(hidden_states, (1, 0, 2))  # [time, batch, d]
        normed = self.layer_norm(hs)
        attn_out = self.self_attention(normed, mask=attention_mask, position_ids=None)
        hs = hs + attn_out
        return mx.transpose(hs, (1, 0, 2))  # [batch, time, d]
