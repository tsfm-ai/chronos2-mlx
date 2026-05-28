"""Rotary position embeddings matching Chronos-2's LLaMA-style RoPE exactly."""
import mlx.core as mx


def apply_rope(
    q: mx.array,
    k: mx.array,
    position_ids: mx.array,
    theta: float = 10000.0,
) -> tuple[mx.array, mx.array]:
    """
    Apply RoPE to q and k.

    q, k:        [batch, heads, seq, d_kv]
    position_ids: [batch, seq] or [1, seq]

    Matches HuggingFace LLaMA RoPE: inv_freq computed with arange(0, dim, 2) / dim.
    """
    d = q.shape[-1]
    # inv_freq: [d//2]
    inv_freq = 1.0 / (theta ** (mx.arange(0, d, 2, dtype=mx.float32) / d))

    batch = position_ids.shape[0]
    seq = position_ids.shape[1]

    # inv_freq_expanded: [batch, d//2, 1]
    inv_freq_exp = mx.broadcast_to(inv_freq[None, :, None], (batch, d // 2, 1))
    # position_ids_expanded: [batch, 1, seq]
    pos_exp = mx.expand_dims(position_ids.astype(mx.float32), 1)

    # freqs: [batch, d//2, seq] -> transpose -> [batch, seq, d//2]
    freqs = mx.transpose(inv_freq_exp @ pos_exp, (0, 2, 1))

    # emb: [batch, seq, d]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos = mx.cos(emb)  # [batch, seq, d]
    sin = mx.sin(emb)

    # Broadcast to [batch, 1, seq, d] for [batch, heads, seq, d_kv]
    cos = mx.expand_dims(cos, 1)
    sin = mx.expand_dims(sin, 1)

    def rotate_half(x: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out.astype(q.dtype), k_out.astype(k.dtype)
