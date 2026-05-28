"""Chronos2MLXPipeline: from_pretrained, predict, predict_df."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from .config import Chronos2MLXConfig
from .model import Chronos2MLXModel
from .weights import load_chronos2_weights, verify_weights_loaded


class Chronos2MLXPipeline:
    def __init__(self, model: Chronos2MLXModel, config: Chronos2MLXConfig):
        self.model = model
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        dtype: str = "float32",
        verify: bool = True,
    ) -> "Chronos2MLXPipeline":
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        repo = Path(snapshot_download(model_id))
        hf_config = json.loads((repo / "config.json").read_text())
        config = Chronos2MLXConfig.from_hf_config(hf_config)
        config.vocab_size = 2 if config.use_reg_token else 1

        model = Chronos2MLXModel(config)

        weights = load_file(str(repo / "model.safetensors"))
        used = load_chronos2_weights(model, weights)

        if verify:
            verify_weights_loaded(model, weights, used)

        if dtype == "bfloat16":
            model.set_dtype(mx.bfloat16)
        elif dtype == "float16":
            model.set_dtype(mx.float16)

        mx.eval(model.parameters())
        return cls(model=model, config=config)

    def predict(
        self,
        context: mx.array | np.ndarray,
        prediction_length: int,
        context_mask: mx.array | np.ndarray | None = None,
        group_ids: mx.array | np.ndarray | None = None,
        future_covariates: mx.array | np.ndarray | None = None,
        future_covariates_mask: mx.array | np.ndarray | None = None,
        quantile_levels: list[float] | None = None,
    ) -> mx.array:
        """
        Low-level tensor predict.

        context: [batch, context_length] — may contain NaN for missing values
        Returns: [batch, num_quantiles, prediction_length]  (subset if quantile_levels given)
        """
        cfg = self.config

        def to_mlx(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return mx.array(x)
            return x

        context = to_mlx(context)
        context_mask = to_mlx(context_mask)
        group_ids = to_mlx(group_ids)
        future_covariates = to_mlx(future_covariates)
        future_covariates_mask = to_mlx(future_covariates_mask)

        num_output_patches = math.ceil(prediction_length / cfg.output_patch_size)
        num_output_patches = min(num_output_patches, cfg.max_output_patches)

        quantile_preds = self.model(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
        )
        mx.eval(quantile_preds)

        # Slice to prediction_length
        quantile_preds = quantile_preds[:, :, :prediction_length]

        if quantile_levels is not None:
            all_q = list(cfg.quantiles)
            indices = [all_q.index(q) for q in quantile_levels]
            quantile_preds = quantile_preds[:, indices, :]

        return quantile_preds

    def embed(
        self,
        context: mx.array | np.ndarray,
        context_mask: mx.array | np.ndarray | None = None,
        group_ids: mx.array | np.ndarray | None = None,
        pooling: str = "reg_token",
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Get encoder embeddings for a batch of time series.

        context: [batch, context_length]
        pooling: one of "reg_token" | "mean_context" | "last_context_patch" | "all"
        Returns:
          embeddings: [batch, d_model] (or [batch, seq, d_model] if pooling="all")
          loc_scale: (loc, scale) tuple from instance normalization
        """
        def to_mlx(x):
            if x is None:
                return None
            return mx.array(x) if isinstance(x, np.ndarray) else x

        context = to_mlx(context)
        context_mask = to_mlx(context_mask)
        group_ids = to_mlx(group_ids)

        # Use 1 output patch (minimum) to get the reg token + future structure
        hidden_states, loc_scale, _, num_context_patches = self.model.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            num_output_patches=1,
        )
        mx.eval(hidden_states)

        # hidden_states: [batch, num_context_patches + 1 + 1, d_model]
        # layout: [ctx_patches..., reg_token, future_patch]
        reg_idx = num_context_patches  # REG token is right after context patches

        if pooling == "reg_token":
            embeddings = hidden_states[:, reg_idx, :]
        elif pooling == "mean_context":
            embeddings = hidden_states[:, :num_context_patches, :].mean(axis=1)
        elif pooling == "last_context_patch":
            embeddings = hidden_states[:, num_context_patches - 1, :]
        elif pooling == "all":
            embeddings = hidden_states
        else:
            raise ValueError(f"Unknown pooling: {pooling!r}. Choose from: reg_token, mean_context, last_context_patch, all")

        return embeddings, loc_scale

    def embed_df(
        self,
        context_df,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        target: str = "target",
        pooling: str = "reg_token",
        context_length: int | None = None,
        batch_size: int = 256,
    ):
        """
        Get encoder embeddings for time series in a long-format DataFrame.

        Returns a DataFrame with columns: id_column, embedding (numpy array).
        """
        import pandas as pd

        max_context = context_length or self.config.context_length
        df = context_df.sort_values([id_column, timestamp_column])
        item_ids = df[id_column].unique().tolist()

        all_embeddings = []
        all_ids = []

        for start in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[start : start + batch_size]
            batch_contexts = []
            for item_id in batch_ids:
                item_df = df[df[id_column] == item_id].sort_values(timestamp_column)
                vals = item_df[target].values[-max_context:]
                arr = np.full(max_context, np.nan, dtype=np.float32)
                arr[-len(vals):] = vals.astype(np.float32)
                batch_contexts.append(arr)

            ctx = mx.array(np.stack(batch_contexts))
            emb, _ = self.embed(ctx, pooling=pooling)
            mx.eval(emb)
            emb_np = np.array(emb.astype(mx.float32))

            for b_idx, item_id in enumerate(batch_ids):
                all_ids.append(item_id)
                all_embeddings.append(emb_np[b_idx])

        return pd.DataFrame({
            id_column: all_ids,
            "embedding": all_embeddings,
        })

    def predict_df(
        self,
        context_df,
        future_df=None,
        prediction_length: int | None = None,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        target: str | list[str] = "target",
        quantile_levels: list[float] | None = None,
        batch_size: int = 256,
        context_length: int | None = None,
    ):
        """Basic predict_df: univariate, single target, no categorical covariates."""
        import pandas as pd

        if prediction_length is None:
            prediction_length = self.config.max_output_patches * self.config.output_patch_size

        if isinstance(target, list):
            target_cols = target
        else:
            target_cols = [target]

        if quantile_levels is None:
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        max_context = context_length or self.config.context_length

        # Sort and group by ID
        df = context_df.sort_values([id_column, timestamp_column])
        item_ids = df[id_column].unique().tolist()

        all_rows = []

        for start_idx in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[start_idx : start_idx + batch_size]
            batch_contexts = []
            batch_future_covs = []
            has_future_covs = False

            for item_id in batch_ids:
                item_df = df[df[id_column] == item_id].sort_values(timestamp_column)
                target_vals = item_df[target_cols[0]].values[-max_context:]
                target_arr = np.full(max_context, np.nan, dtype=np.float32)
                target_arr[-len(target_vals):] = target_vals.astype(np.float32)
                batch_contexts.append(target_arr)

                if future_df is not None:
                    future_item = future_df[future_df[id_column] == item_id].sort_values(timestamp_column)
                    covariate_cols = [c for c in future_item.columns if c not in [id_column, timestamp_column] + target_cols]
                    if covariate_cols:
                        has_future_covs = True
                        cov_vals = future_item[covariate_cols[0]].values[:prediction_length]
                        cov_arr = np.full(prediction_length, np.nan, dtype=np.float32)
                        cov_arr[:len(cov_vals)] = cov_vals.astype(np.float32)
                        batch_future_covs.append(cov_arr)

            ctx = mx.array(np.stack(batch_contexts))
            fut_cov = mx.array(np.stack(batch_future_covs)) if batch_future_covs else None

            preds = self.predict(
                context=ctx,
                prediction_length=prediction_length,
                future_covariates=fut_cov,
                quantile_levels=quantile_levels,
            )

            preds_np = np.array(preds)  # [batch, num_q, horizon]

            for b_idx, item_id in enumerate(batch_ids):
                item_df = df[df[id_column] == item_id].sort_values(timestamp_column)
                last_ts = item_df[timestamp_column].iloc[-1]
                try:
                    freq = pd.infer_freq(item_df[timestamp_column])
                except Exception:
                    freq = None
                if freq:
                    future_timestamps = pd.date_range(start=last_ts, periods=prediction_length + 1, freq=freq)[1:]
                else:
                    future_timestamps = [None] * prediction_length

                median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else 0
                median_preds = preds_np[b_idx, median_idx]

                row_data = {
                    id_column: [item_id] * prediction_length,
                    timestamp_column: list(future_timestamps),
                    "target_name": [target_cols[0]] * prediction_length,
                    "predictions": median_preds.tolist(),
                }
                for q_idx, q in enumerate(quantile_levels):
                    row_data[str(q)] = preds_np[b_idx, q_idx].tolist()

                all_rows.append(pd.DataFrame(row_data))

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()
