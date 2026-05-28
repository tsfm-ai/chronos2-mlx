"""Fine-tuning loop for chronos2-mlx: head-only, LoRA, and full fine-tune."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .adapters import LoRAConfig, apply_lora
from .losses import pinball_loss
from .model import Chronos2MLXModel
from .preprocessing import InstanceNorm


@dataclass
class TrainConfig:
    prediction_length: int = 24
    context_length: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_steps: int = 1000
    warmup_steps: int = 100
    log_every: int = 50
    eval_every: int = 200
    grad_clip: float = 1.0
    finetune_mode: str = "lora"  # "lora" | "head" | "full"
    lora: LoRAConfig = field(default_factory=LoRAConfig)


def _rolling_windows(
    series: np.ndarray,
    context_length: int,
    prediction_length: int,
    min_past: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (context, target) pairs from a time series."""
    if min_past is None:
        min_past = prediction_length
    total = len(series)
    for end in range(min_past + prediction_length, total + 1):
        start = max(0, end - prediction_length - context_length)
        ctx = series[start : end - prediction_length]
        tgt = series[end - prediction_length : end]
        if len(ctx) >= min_past:
            # Pad context to context_length
            padded = np.full(context_length, np.nan, dtype=np.float32)
            padded[-len(ctx):] = ctx
            yield padded, tgt.astype(np.float32)


class TimeSeriesDataLoader:
    """Simple rolling-window data loader for fine-tuning."""

    def __init__(
        self,
        series_list: list[np.ndarray],
        context_length: int,
        prediction_length: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Precompute all windows
        windows = []
        for series in series_list:
            for ctx, tgt in _rolling_windows(series, context_length, prediction_length):
                windows.append((ctx, tgt))

        if not windows:
            raise ValueError("No valid windows found. Check context_length and prediction_length vs data length.")

        self.contexts = np.stack([w[0] for w in windows])
        self.targets = np.stack([w[1] for w in windows])

    def __len__(self):
        return math.ceil(len(self.contexts) / self.batch_size)

    def __iter__(self):
        n = len(self.contexts)
        indices = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = indices[start : start + self.batch_size]
            yield (
                mx.array(self.contexts[idx]),
                mx.array(self.targets[idx]),
            )


def _setup_finetune_mode(model: Chronos2MLXModel, config: TrainConfig) -> None:
    if config.finetune_mode == "full":
        model.unfreeze()
    elif config.finetune_mode == "head":
        model.freeze()
        model.output_patch_embedding.unfreeze()
        model.encoder.final_layer_norm.unfreeze()
    elif config.finetune_mode == "lora":
        model.freeze()
        from .adapters import _get_module, _all_lora_paths, LoRALinear

        # Apply if not already applied
        existing = _all_lora_paths(model)
        if not existing:
            injected = apply_lora(model, config.lora)
            if not injected:
                raise RuntimeError(
                    f"No LoRA adapters injected. Check target_projections={config.lora.target_projections!r} "
                    f"and target_attention_layers={config.lora.target_attention_layers!r}."
                )
            existing = injected

        # Unfreeze only lora_a / lora_b — re-freeze base after
        for path in existing:
            module = _get_module(model, path)
            if isinstance(module, LoRALinear):
                module.unfreeze()
                module.base.freeze()
    else:
        raise ValueError(f"Unknown finetune_mode: {config.finetune_mode!r}")


def fine_tune(
    model: Chronos2MLXModel,
    train_series: list[np.ndarray],
    config: TrainConfig,
    val_series: list[np.ndarray] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Fine-tune a Chronos-2 MLX model.

    train_series: list of 1-D numpy arrays (raw, unnormalized time series)
    Returns: training log (list of dicts with step, loss, etc.)
    """
    _setup_finetune_mode(model, config)

    n_trainable = sum(
        p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters())
    )
    n_total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    if verbose:
        print(f"Fine-tuning: {config.finetune_mode} mode | {n_trainable:,} trainable / {n_total:,} total params")

    train_loader = TimeSeriesDataLoader(
        train_series, config.context_length, config.prediction_length, config.batch_size
    )
    val_loader = TimeSeriesDataLoader(
        val_series, config.context_length, config.prediction_length, config.batch_size
    ) if val_series else None

    optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    instance_norm = model.instance_norm

    def loss_fn(mdl: Chronos2MLXModel, context: mx.array, future_target: mx.array) -> mx.array:
        num_patches = math.ceil(config.prediction_length / mdl.config.output_patch_size)
        hidden_states, loc_scale, _, _ = mdl.encode(context=context, num_output_patches=num_patches)
        forecast_embeds = hidden_states[:, -num_patches:]
        quantile_preds = mdl.output_patch_embedding(forecast_embeds)

        q = mdl.config.num_quantiles
        p = mdl.config.output_patch_size
        n = num_patches
        batch = context.shape[0]
        quantile_preds = mx.reshape(quantile_preds, (batch, n, q, p))
        quantile_preds = mx.transpose(quantile_preds, (0, 2, 1, 3))
        quantile_preds = mx.reshape(quantile_preds, (batch, q, n * p))

        # Normalize future_target the same way instance_norm normalized context
        norm_target, _ = instance_norm(future_target, loc_scale)
        norm_target = norm_target[:, : config.prediction_length]

        mask = (~mx.isnan(future_target)).astype(quantile_preds.dtype)[:, : config.prediction_length]

        return pinball_loss(
            pred=quantile_preds[:, :, : config.prediction_length],
            target=norm_target,
            quantiles=mdl.quantiles,
            mask=mask,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    log = []
    step = 0

    while step < config.max_steps:
        for context, future_target in train_loader:
            if step >= config.max_steps:
                break

            loss, grads = loss_and_grad(model, context, future_target)

            # Gradient clipping
            if config.grad_clip > 0:
                grads, _ = optim.clip_grad_norm(grads, config.grad_clip)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_val = float(loss)
            step += 1

            if verbose and step % config.log_every == 0:
                print(f"  step={step:4d}/{config.max_steps}  loss={loss_val:.4f}")

            entry = {"step": step, "train_loss": loss_val}

            if val_loader and step % config.eval_every == 0:
                val_losses = []
                for val_ctx, val_tgt in val_loader:
                    v_loss = loss_fn(model, val_ctx, val_tgt)
                    mx.eval(v_loss)
                    val_losses.append(float(v_loss))
                entry["val_loss"] = float(np.mean(val_losses))
                if verbose:
                    print(f"         val_loss={entry['val_loss']:.4f}")

            log.append(entry)

    return log
