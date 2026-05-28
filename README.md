# chronos2-mlx

[![CI](https://github.com/tsfm-ai/chronos2-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/tsfm-ai/chronos2-mlx/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/chronos2-mlx)](https://pypi.org/project/chronos2-mlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**MLX inference backend for [Chronos-2](https://github.com/amazon-science/chronos-forecasting) on Apple Silicon.**

Runs `amazon/chronos-2` natively on M-series chips using [MLX](https://github.com/ml-explore/mlx) — no PyTorch dependency for inference. Supports fp32, bf16, int8, and int4 weight-only quantization, LoRA fine-tuning, QLoRA, and the full Chronos-2 feature set including group attention and future covariates.

---

## Requirements

- macOS on Apple Silicon (M1 or later)
- Python ≥ 3.10
- MLX ≥ 0.31

## Install

```bash
pip install chronos2-mlx
```

## Quickstart

```python
import numpy as np
import mlx.core as mx
from chronos2_mlx import Chronos2MLXPipeline

pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")

# Context window — shape [batch, time]
context = mx.array(np.random.standard_normal((1, 512)).astype(np.float32))

# Returns [batch, num_quantiles, horizon]
quantiles = pipe.predict(context, prediction_length=24)
mx.eval(quantiles)

# Quantile levels: 0.1, 0.2, ..., 0.9 plus finer tails (21 total)
q_levels = np.array(pipe.model.quantiles.astype(mx.float32))
median = np.array(quantiles[0, np.argmin(np.abs(q_levels - 0.5))].astype(mx.float32))
```

### DataFrame API

```python
import pandas as pd
from chronos2_mlx import Chronos2MLXPipeline

pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")

df = pd.DataFrame({
    "item_id": ["A"] * 100 + ["B"] * 100,
    "ds": pd.date_range("2023-01-01", periods=100).tolist() * 2,
    "y": np.random.standard_normal(200).tolist(),
})

forecast_df = pipe.predict_df(df, prediction_length=24)
```

---

## Features

- **Exact parity with PyTorch Chronos-2** — verified layer-by-layer; max absolute error < 6e-6 in fp32
- **bf16 inference** — ~1.5× speedup, < 0.3% mean relative error vs fp32
- **int8 / int4 quantization** — weight-only, group-size 64; int8 MRE < 0.4%, int4 MRE ~6–7%
- **Group attention** — batch multiple related series together and the encoder shares cross-series information automatically
- **LoRA fine-tuning** — inject adapters onto any attention projection (`q`, `k`, `v`, `o`), any subset of layers
- **QLoRA** — LoRA on top of int4/int8 quantized base weights
- **Adapter save/load/fuse** — adapters stored as `.npz`; fuse into base weights for zero-overhead serving
- **Head-only and full fine-tuning** — in addition to LoRA
- **Future covariates** — pass known-future features for conditional forecasting
- **Embeddings** — `embed()` and `embed_df()` extract contextual time-series representations
- **`mx.compile` compatible** — wrap `pipe.predict` for additional throughput

---

## Precision and quantization

| Mode | MRE vs fp32 | Memory (120M model) |
|------|-------------|---------------------|
| fp32 | baseline | ~480 MB |
| bf16 | < 0.3% | ~240 MB |
| int8, gs=64 | < 0.5% | ~120 MB |
| int4, gs=64 | ~6–7% | ~60 MB |

```python
from chronos2_mlx import Chronos2MLXPipeline, quantize_model, param_footprint

# bf16
pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2", dtype=mx.bfloat16)

# int8
pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
quantize_model(pipe.model, bits=8, group_size=64)
print(param_footprint(pipe.model))  # {'total_params': ..., 'total_bytes': ...}
```

---

## Fine-tuning

### LoRA

```python
from chronos2_mlx import (
    Chronos2MLXPipeline, LoRAConfig, TrainConfig,
    apply_lora, fine_tune, save_adapter, load_adapter, fuse_lora,
)

pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")

cfg = LoRAConfig(
    rank=8,
    alpha=16.0,
    target_projections=["q", "v"],       # which attention projections
    target_attention_layers=[0, 1],      # 0=time attention, 1=group attention
)
apply_lora(pipe.model, cfg)

train_cfg = TrainConfig(
    finetune_mode="lora",
    lora=cfg,
    prediction_length=24,
    context_length=512,
    learning_rate=1e-4,
    max_steps=500,
    batch_size=16,
)

series = [my_numpy_array_1, my_numpy_array_2]  # list of 1-D np.ndarray
log = fine_tune(pipe.model, series, train_cfg, verbose=True)

# Save adapter only (base weights unchanged)
save_adapter(pipe.model, "my_adapter/", cfg)

# Load on a fresh model
pipe2 = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
load_adapter(pipe2.model, "my_adapter/")

# Or fuse into base weights for zero-overhead serving
fuse_lora(pipe.model)
```

### QLoRA

```python
from chronos2_mlx import quantize_model, apply_lora, LoRAConfig

pipe = Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")
quantize_model(pipe.model, bits=4, group_size=64)   # quantize base

cfg = LoRAConfig(rank=4, target_projections=["q", "v"])
apply_lora(pipe.model, cfg)                          # LoRA wraps QuantizedLinear
```

### Head-only and full fine-tuning

```python
train_cfg = TrainConfig(finetune_mode="head")   # output head + final norm only
train_cfg = TrainConfig(finetune_mode="full")   # all parameters
```

---

## Group attention

Pass a batch of related series and Chronos-2's cross-series attention fires automatically:

```python
# All series encoded jointly — encodings attend across the batch
context = mx.array(np.stack([zone_a, zone_b, zone_c]))  # [3, context_len]
quantiles = pipe.predict(context, prediction_length=24)
```

For independent univariate forecasts, pass each series in its own batch.

---

## Embeddings

```python
# Returns [batch, d_model] contextual embeddings (REG token)
embeddings = pipe.embed(context)
mx.eval(embeddings)

# DataFrame variant
emb_df = pipe.embed_df(df, id_col="item_id", time_col="ds", target_col="y")
```

---

## API reference

### `Chronos2MLXPipeline`

| Method | Description |
|--------|-------------|
| `from_pretrained(model_id, dtype=mx.float32)` | Load from HuggingFace Hub |
| `predict(context, prediction_length, ...)` | Returns `[batch, num_quantiles, horizon]` |
| `predict_df(df, prediction_length, ...)` | DataFrame forecast |
| `embed(context, pooling="reg")` | Contextual embeddings |
| `embed_df(df, ...)` | DataFrame embeddings |

### `LoRAConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `rank` | 8 | LoRA rank |
| `alpha` | 16.0 | Scaling factor (effective lr = alpha/rank) |
| `dropout` | 0.0 | Dropout on LoRA path |
| `target_projections` | `["q", "v"]` | Attention projections to adapt |
| `target_attention_layers` | `[0, 1]` | 0=time attention, 1=group attention |

### `TrainConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `finetune_mode` | `"lora"` | `"lora"` / `"head"` / `"full"` |
| `prediction_length` | 24 | Forecast horizon |
| `context_length` | 512 | Maximum context window |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `weight_decay` | 0.01 | AdamW weight decay |
| `batch_size` | 32 | Training batch size |
| `max_steps` | 1000 | Total training steps |
| `grad_clip` | 1.0 | Gradient norm clip |

### `QuantizeConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `bits` | 8 | Quantization bits (4 or 8) |
| `group_size` | 64 | Quantization group size |

---

## Development

```bash
git clone https://github.com/tsfm-ai/chronos2-mlx
cd chronos2-mlx
pip install -e ".[dev]"

# Run tests (fast — no torch required)
pytest -m "not parity"

# Run parity tests (requires chronos-forecasting + torch)
pytest -m parity

# Run all
pytest
```

---

## License

Apache-2.0. Chronos-2 model weights are subject to [Amazon's license](https://huggingface.co/amazon/chronos-2).
