"""Feed-forward, residual blocks, and encoder blocks."""
import mlx.core as mx
import mlx.nn as nn

from .attention import GroupSelfAttention, TimeSelfAttention
from .config import Chronos2MLXConfig
from .norm import RMSNorm


class FeedForward(nn.Module):
    def __init__(self, config: Chronos2MLXConfig):
        super().__init__()
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        normed = self.layer_norm(x)
        return x + self.wo(nn.relu(self.wi(normed)))


class ResidualBlock(nn.Module):
    """Matches chronos2 ResidualBlock: hidden_layer + act + output_layer + residual_layer skip."""

    def __init__(self, in_dim: int, h_dim: int, out_dim: int):
        super().__init__()
        # All three layers have bias (weights have bias entries in safetensors)
        self.hidden_layer = nn.Linear(in_dim, h_dim, bias=True)
        self.output_layer = nn.Linear(h_dim, out_dim, bias=True)
        self.residual_layer = nn.Linear(in_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        hid = nn.relu(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        return out + res


class Chronos2EncoderBlock(nn.Module):
    def __init__(self, config: Chronos2MLXConfig):
        super().__init__()
        # Match PyTorch: layer[0]=TimeSelfAttention, layer[1]=GroupSelfAttention, layer[2]=FeedForward
        self.layer = [
            TimeSelfAttention(config),
            GroupSelfAttention(config),
            FeedForward(config),
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        position_ids: mx.array,
        attention_mask: mx.array | None = None,
        group_time_mask: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.layer[0](hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = self.layer[1](hidden_states, attention_mask=group_time_mask)
        hidden_states = self.layer[2](hidden_states)
        return hidden_states


class Chronos2Encoder(nn.Module):
    def __init__(self, config: Chronos2MLXConfig):
        super().__init__()
        self.block = [Chronos2EncoderBlock(config) for _ in range(config.num_layers)]
        self.final_layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    @staticmethod
    def _make_time_mask(attention_mask: mx.array, dtype: mx.Dtype) -> mx.array:
        """[batch, seq] binary -> [batch, 1, 1, seq] additive float mask."""
        mask = attention_mask[:, None, None, :].astype(mx.float32)
        neg_inf = mx.array(mx.finfo(mx.float32).min, dtype=mx.float32)
        mask = (1.0 - mask) * neg_inf
        return mask.astype(dtype)

    @staticmethod
    def _make_group_time_mask(
        group_ids: mx.array,
        attention_mask: mx.array,
        dtype: mx.Dtype,
    ) -> mx.array:
        """
        Produces group_time_mask shaped [time, 1, batch, batch].

        Matches PyTorch:
            group_mask = group_ids[:, None] == group_ids[None, :]  # [batch, batch]
            group_time_mask = einsum("qb, bt -> qbt", group_mask, attention_mask)  # [batch, batch, time]
            group_time_mask = rearrange("q b t -> t 1 q b")
            group_time_mask = (1 - group_time_mask) * -inf
        """
        batch = group_ids.shape[0]
        # [batch, batch] — True where same group
        group_mask = (mx.expand_dims(group_ids, 1) == mx.expand_dims(group_ids, 0)).astype(mx.float32)
        # attention_mask: [batch, seq] — 1=valid
        # einsum "qb, bt -> qbt": outer product of group_mask and attention_mask
        # group_mask: [batch_q, batch_k], attention_mask: [batch_k, time]
        # result: [batch_q, batch_k, time] — valid if same group AND time is valid
        seq = attention_mask.shape[1]
        # manual einsum via broadcasting: [batch_q, batch_k, 1] * [1, batch_k, time]
        gtm = mx.expand_dims(group_mask, -1) * mx.expand_dims(attention_mask, 0)  # [batch_q, batch_k, time]
        # rearrange "q b t -> t 1 q b": [time, 1, batch_q, batch_k]
        gtm = mx.transpose(gtm, (2, 0, 1))   # [time, batch_q, batch_k]
        gtm = mx.expand_dims(gtm, 1)          # [time, 1, batch_q, batch_k]
        neg_inf = mx.array(mx.finfo(mx.float32).min, dtype=mx.float32)
        gtm = ((1.0 - gtm.astype(mx.float32)) * neg_inf).astype(dtype)
        return gtm

    def __call__(
        self,
        inputs_embeds: mx.array,
        group_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_length, _ = inputs_embeds.shape

        if position_ids is None:
            position_ids = mx.expand_dims(mx.arange(seq_length, dtype=mx.int32), 0)  # [1, seq]
            position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_length), dtype=inputs_embeds.dtype)

        dtype = inputs_embeds.dtype
        extended_attn_mask = self._make_time_mask(attention_mask, dtype)
        group_time_mask = self._make_group_time_mask(group_ids, attention_mask, dtype)

        hidden_states = inputs_embeds

        for block in self.block:
            hidden_states = block(
                hidden_states,
                position_ids=position_ids,
                attention_mask=extended_attn_mask,
                group_time_mask=group_time_mask,
            )

        return self.final_layer_norm(hidden_states)
