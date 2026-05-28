"""Chronos2MLXModel: full forward pass matching PyTorch Chronos-2 exactly."""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import Chronos2MLXConfig
from .layers import Chronos2Encoder, ResidualBlock
from .preprocessing import InstanceNorm, patch_sequence


class Chronos2MLXModel(nn.Module):
    def __init__(self, config: Chronos2MLXConfig):
        super().__init__()
        self.config = config

        # vocab_size=2: token 0=PAD, token 1=REG
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Input patch embedding: 3 * patch_size features (time_enc, values, mask)
        self.input_patch_embedding = ResidualBlock(
            in_dim=config.input_patch_size * 3,
            h_dim=config.d_ff,
            out_dim=config.d_model,
        )

        # Output patch embedding: d_model -> output_patch_size * num_quantiles
        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=config.output_patch_size * config.num_quantiles,
        )

        self.encoder = Chronos2Encoder(config)
        self.instance_norm = InstanceNorm(use_arcsinh=config.use_arcsinh)

        # Register quantiles as a non-trainable buffer
        self._quantiles = mx.array(list(config.quantiles), dtype=mx.float32)

    @property
    def quantiles(self) -> mx.array:
        return self._quantiles

    def _prepare_patched_context(
        self,
        context: mx.array,
        context_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, tuple[mx.array, mx.array]]:
        """Normalize, patch, and build time encoding for context."""
        cfg = self.config

        if context_mask is not None:
            context_mask = context_mask.astype(context.dtype)
        else:
            context_mask = (~mx.isnan(context)).astype(context.dtype)

        batch_size = context.shape[0]
        context_length = context.shape[1]

        # Truncate if longer than model context
        if context_length > cfg.context_length:
            context = context[:, -cfg.context_length:]
            context_mask = context_mask[:, -cfg.context_length:]

        # Instance normalization (in float32)
        context, loc_scale = self.instance_norm(context)
        context = context.astype(self.dtype)
        context_mask = context_mask.astype(self.dtype)

        # Patch
        patched_context = patch_sequence(context, cfg.input_patch_size, cfg.input_patch_stride)
        patched_mask = patch_sequence(context_mask, cfg.input_patch_size, cfg.input_patch_stride)
        patched_mask = mx.nan_to_num(patched_mask, nan=0.0)
        patched_context = mx.where(patched_mask > 0.0, patched_context, 0.0)

        num_context_patches = patched_context.shape[1]

        # Attention mask: 1 if any element in patch is observed
        attention_mask = (patched_mask.sum(axis=-1) > 0).astype(self.dtype)  # [batch, num_context_patches]

        # Context time encoding: [-C, ..., -1] / time_encoding_scale
        final_context_length = num_context_patches * cfg.input_patch_size
        ctx_time = mx.arange(-final_context_length, 0, dtype=mx.float32) / cfg.time_encoding_scale
        # [final_context_length] -> [batch, num_context_patches, patch_size]
        ctx_time = mx.reshape(ctx_time, (1, num_context_patches, cfg.input_patch_size))
        ctx_time = mx.broadcast_to(ctx_time, (batch_size, num_context_patches, cfg.input_patch_size))
        ctx_time = ctx_time.astype(self.dtype)

        # Concatenate [time_enc, values, mask] along last dim -> [batch, num_patches, 3*patch_size]
        patched_context = mx.concatenate([ctx_time, patched_context, patched_mask], axis=-1)

        return patched_context, attention_mask, loc_scale

    def _prepare_patched_future(
        self,
        future_covariates: mx.array | None,
        future_covariates_mask: mx.array | None,
        loc_scale: tuple[mx.array, mx.array],
        num_output_patches: int,
        batch_size: int,
    ) -> tuple[mx.array, mx.array]:
        """Build future patches with time encoding and optional covariates."""
        cfg = self.config
        output_patch_size = cfg.output_patch_size

        if future_covariates is not None:
            future_covariates, _ = self.instance_norm(future_covariates, loc_scale)
            future_covariates = future_covariates.astype(self.dtype)

            if future_covariates_mask is None:
                future_covariates_mask = (~mx.isnan(future_covariates)).astype(future_covariates.dtype)

            future_covariates = mx.where(future_covariates_mask > 0.0, future_covariates, 0.0)

            # Pad to num_output_patches * output_patch_size if needed
            target_len = num_output_patches * output_patch_size
            current_len = future_covariates.shape[-1]
            if target_len > current_len:
                pad_shape = (*future_covariates.shape[:-1], target_len - current_len)
                future_covariates = mx.concatenate(
                    [future_covariates, mx.zeros(pad_shape, dtype=future_covariates.dtype)], axis=-1
                )
                future_covariates_mask = mx.concatenate(
                    [future_covariates_mask, mx.zeros(pad_shape, dtype=future_covariates_mask.dtype)], axis=-1
                )

            # Reshape to patches
            patched_future_cov = mx.reshape(future_covariates, (batch_size, num_output_patches, output_patch_size))
            patched_future_cov_mask = mx.reshape(
                future_covariates_mask, (batch_size, num_output_patches, output_patch_size)
            )
        else:
            patched_future_cov = mx.zeros((batch_size, num_output_patches, output_patch_size), dtype=self.dtype)
            patched_future_cov_mask = mx.zeros((batch_size, num_output_patches, output_patch_size), dtype=self.dtype)

        # Future time encoding: [0, ..., H-1] / time_encoding_scale
        final_future_length = num_output_patches * output_patch_size
        fut_time = mx.arange(0, final_future_length, dtype=mx.float32) / cfg.time_encoding_scale
        fut_time = mx.reshape(fut_time, (1, num_output_patches, output_patch_size))
        fut_time = mx.broadcast_to(fut_time, (batch_size, num_output_patches, output_patch_size))
        fut_time = fut_time.astype(self.dtype)

        # [time_enc, covariates, mask]
        patched_future = mx.concatenate([fut_time, patched_future_cov, patched_future_cov_mask], axis=-1)

        return patched_future, patched_future_cov_mask

    @property
    def dtype(self) -> mx.Dtype:
        return self.shared.weight.dtype

    def encode(
        self,
        context: mx.array,
        context_mask: mx.array | None = None,
        group_ids: mx.array | None = None,
        future_covariates: mx.array | None = None,
        future_covariates_mask: mx.array | None = None,
        num_output_patches: int = 1,
    ) -> tuple[mx.array, tuple[mx.array, mx.array], mx.array, int]:
        batch_size = context.shape[0]

        patched_context, attention_mask, loc_scale = self._prepare_patched_context(context, context_mask)
        num_context_patches = attention_mask.shape[-1]

        # Embed context patches: [batch, num_context_patches, d_model]
        input_embeds: mx.array = self.input_patch_embedding(patched_context)

        # Append [REG] token
        if self.config.use_reg_token:
            reg_ids = mx.full((batch_size, 1), self.config.reg_token_id, dtype=mx.int32)
            reg_embeds = self.shared(reg_ids)  # [batch, 1, d_model]
            input_embeds = mx.concatenate([input_embeds, reg_embeds], axis=1)
            attention_mask = mx.concatenate(
                [attention_mask, mx.ones((batch_size, 1), dtype=attention_mask.dtype)], axis=1
            )

        patched_future, patched_future_cov_mask = self._prepare_patched_future(
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            loc_scale=loc_scale,
            num_output_patches=num_output_patches,
            batch_size=batch_size,
        )
        future_attn_mask = mx.ones((batch_size, num_output_patches), dtype=attention_mask.dtype)

        # Embed future patches: [batch, num_output_patches, d_model]
        future_embeds: mx.array = self.input_patch_embedding(patched_future)

        # Concatenate: [context + reg + future]
        input_embeds = mx.concatenate([input_embeds, future_embeds], axis=1)
        attention_mask = mx.concatenate([attention_mask, future_attn_mask], axis=1)

        if group_ids is None:
            group_ids = mx.arange(batch_size, dtype=mx.int32)

        hidden_states = self.encoder(
            inputs_embeds=input_embeds,
            group_ids=group_ids,
            attention_mask=attention_mask,
        )

        return hidden_states, loc_scale, patched_future_cov_mask, num_context_patches

    def __call__(
        self,
        context: mx.array,
        context_mask: mx.array | None = None,
        group_ids: mx.array | None = None,
        future_covariates: mx.array | None = None,
        future_covariates_mask: mx.array | None = None,
        num_output_patches: int = 1,
    ) -> mx.array:
        """
        Forward pass.
        Returns quantile_preds: [batch, num_quantiles, num_output_patches * output_patch_size]
        already de-normalized.
        """
        cfg = self.config
        batch_size = context.shape[0]

        hidden_states, loc_scale, patched_future_cov_mask, num_context_patches = self.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
        )

        # Slice last num_output_patches positions
        forecast_embeds = hidden_states[:, -num_output_patches:]  # [batch, num_output_patches, d_model]

        # Output head: [batch, num_output_patches, output_patch_size * num_quantiles]
        quantile_preds = self.output_patch_embedding(forecast_embeds)

        # Rearrange: [batch, num_output_patches, q*p] -> [batch, q, n*p]
        q = cfg.num_quantiles
        p = cfg.output_patch_size
        n = num_output_patches
        quantile_preds = mx.reshape(quantile_preds, (batch_size, n, q, p))
        quantile_preds = mx.transpose(quantile_preds, (0, 2, 1, 3))  # [batch, q, n, p]
        quantile_preds = mx.reshape(quantile_preds, (batch_size, q, n * p))

        # Denormalize: flatten to [batch, q*h], inverse, reshape back
        h = n * p
        flat = mx.reshape(quantile_preds, (batch_size, q * h))
        flat = self.instance_norm.inverse(flat, loc_scale)
        quantile_preds = mx.reshape(flat, (batch_size, q, h))

        return quantile_preds  # [batch, num_quantiles, horizon]
