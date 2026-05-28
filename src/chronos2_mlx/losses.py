"""Pinball (quantile) loss matching Chronos-2 training objective."""
import mlx.core as mx


def pinball_loss(
    pred: mx.array,
    target: mx.array,
    quantiles: mx.array | list[float],
    mask: mx.array | None = None,
) -> mx.array:
    """
    Pinball loss matching Chronos-2's _compute_loss exactly.

    pred:      [batch, num_quantiles, horizon]
    target:    [batch, horizon] — normalized target values
    quantiles: [num_quantiles]
    mask:      [batch, horizon] — 1 for valid prediction targets, 0 for masked/covariate

    Loss is: mean over horizon, sum over quantiles, mean over batch.
    """
    if not isinstance(quantiles, mx.array):
        quantiles = mx.array(list(quantiles), dtype=mx.float32)

    # target: [batch, 1, horizon] for broadcasting
    target_exp = mx.expand_dims(target, 1)
    # quantiles: [1, num_q, 1]
    q = quantiles.reshape(1, -1, 1)

    error = target_exp - pred
    # Pinball: 2 * |error| * |indicator - q|
    indicator = (target_exp <= pred).astype(pred.dtype)
    loss = 2.0 * mx.abs(error * (indicator - q))  # [batch, num_q, horizon]

    if mask is not None:
        mask_exp = mx.expand_dims(mask, 1)  # [batch, 1, horizon]
        loss = loss * mask_exp

    # Mean over horizon, sum over quantiles, mean over batch
    return loss.mean(axis=-1).sum(axis=-1).mean()
