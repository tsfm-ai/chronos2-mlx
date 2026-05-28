"""Instance normalization and patching — exact matches to Chronos-2 chronos_bolt implementations."""
import mlx.core as mx
import numpy as np


class InstanceNorm:
    """
    Standardize along last dim, optionally apply arcsinh.
    Matches chronos_bolt.InstanceNorm exactly.
    """

    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = False):
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def __call__(
        self,
        x: mx.array,
        loc_scale: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        orig_dtype = x.dtype
        x32 = x.astype(mx.float32)

        if loc_scale is None:
            # nanmean and nanstd — NaN values are excluded
            # MLX doesn't have nanmean, so we use observed mask
            nan_mask = mx.isnan(x32)
            observed = mx.where(nan_mask, mx.zeros_like(x32), x32)
            count = mx.sum(~nan_mask, axis=-1, keepdims=True).astype(mx.float32)
            count = mx.maximum(count, 1.0)
            loc = mx.sum(observed, axis=-1, keepdims=True) / count

            centered = mx.where(nan_mask, mx.zeros_like(x32), x32 - loc)
            var = mx.sum(centered * centered, axis=-1, keepdims=True) / count
            scale = mx.sqrt(var)
            # Replicate nan_to_num behavior: nan -> 0 for loc, nan -> 1 for scale
            loc = mx.where(mx.isnan(loc), mx.zeros_like(loc), loc)
            scale = mx.where(mx.isnan(scale), mx.ones_like(scale), scale)
            scale = mx.where(scale == 0.0, mx.full(scale.shape, self.eps), scale)
        else:
            loc, scale = loc_scale

        scaled = (x32 - loc) / scale

        if self.use_arcsinh:
            scaled = mx.arcsinh(scaled)

        return scaled.astype(orig_dtype), (loc, scale)

    def inverse(
        self,
        x: mx.array,
        loc_scale: tuple[mx.array, mx.array],
    ) -> mx.array:
        orig_dtype = x.dtype
        x32 = x.astype(mx.float32)
        loc, scale = loc_scale

        if self.use_arcsinh:
            x32 = mx.sinh(x32)

        result = x32 * scale + loc
        return result.astype(orig_dtype)


def patch_sequence(x: mx.array, patch_size: int, patch_stride: int) -> mx.array:
    """
    Matches chronos_bolt.Patch: left-pad with NaN if not divisible, then unfold.

    x: [batch, time]
    returns: [batch, num_patches, patch_size]
    """
    length = x.shape[-1]
    if length % patch_size != 0:
        pad_len = patch_size - (length % patch_size)
        pad_shape = (*x.shape[:-1], pad_len)
        padding = mx.full(pad_shape, float("nan"), dtype=x.dtype)
        x = mx.concatenate([padding, x], axis=-1)

    # unfold: non-overlapping since stride == patch_size
    length = x.shape[-1]
    num_patches = (length - patch_size) // patch_stride + 1
    patches = mx.stack(
        [x[..., i * patch_stride : i * patch_stride + patch_size] for i in range(num_patches)],
        axis=-2,
    )
    return patches  # [batch, num_patches, patch_size]
