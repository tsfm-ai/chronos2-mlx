"""Verify all HF weights load without gaps or extras."""
import pytest
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from pathlib import Path

from chronos2_mlx import Chronos2MLXConfig, Chronos2MLXModel
from chronos2_mlx.weights import load_chronos2_weights


@pytest.fixture(scope="module")
def hf_weights():
    repo = Path(snapshot_download("amazon/chronos-2"))
    return load_file(str(repo / "model.safetensors"))


@pytest.fixture(scope="module")
def fresh_model():
    import json
    from huggingface_hub import snapshot_download
    repo = Path(snapshot_download("amazon/chronos-2"))
    hf_config = json.loads((repo / "config.json").read_text())
    cfg = Chronos2MLXConfig.from_hf_config(hf_config)
    cfg.vocab_size = 2
    return Chronos2MLXModel(cfg)


def test_all_weights_consumed(fresh_model, hf_weights):
    used = load_chronos2_weights(fresh_model, hf_weights)
    unused = set(hf_weights.keys()) - used
    assert unused == set(), f"Weights not loaded: {unused}"


def test_no_extra_weights(fresh_model, hf_weights):
    used = load_chronos2_weights(fresh_model, hf_weights)
    nonexistent = used - set(hf_weights.keys())
    assert nonexistent == set(), f"Tried to load nonexistent keys: {nonexistent}"


def test_weight_count():
    """Chronos-2 has known parameter count (≈120M)."""
    import mlx.core as mx
    import json
    from huggingface_hub import snapshot_download
    repo = Path(snapshot_download("amazon/chronos-2"))
    hf_config = json.loads((repo / "config.json").read_text())
    cfg = Chronos2MLXConfig.from_hf_config(hf_config)
    cfg.vocab_size = 2
    model = Chronos2MLXModel(cfg)

    from safetensors.torch import load_file
    weights = load_file(str(repo / "model.safetensors"))
    load_chronos2_weights(model, weights)
    mx.eval(model.parameters())

    import mlx.nn as nn
    flat = nn.utils.tree_flatten(model.parameters())
    total = sum(v.size for _, v in flat)
    # Chronos-2 is ~120M parameters; allow ±10%
    assert 100_000_000 < total < 140_000_000, f"Unexpected param count: {total:,}"
