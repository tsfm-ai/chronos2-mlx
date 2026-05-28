"""Config loading tests."""
import pytest
from chronos2_mlx import Chronos2MLXConfig


def test_defaults():
    cfg = Chronos2MLXConfig()
    assert cfg.d_model == 768
    assert cfg.num_layers == 12
    assert cfg.num_heads == 12
    assert cfg.input_patch_size == 16
    assert cfg.num_quantiles == 21


def test_from_hf_config():
    hf = {
        "chronos_config": {
            "context_length": 8192,
            "input_patch_size": 16,
            "input_patch_stride": 16,
            "max_output_patches": 64,
            "output_patch_size": 16,
            "quantiles": [0.1, 0.5, 0.9],
            "use_arcsinh": True,
            "use_reg_token": True,
            "time_encoding_scale": 8192,
        },
        "d_model": 768,
        "d_kv": 64,
        "d_ff": 3072,
        "num_heads": 12,
        "num_layers": 12,
        "layer_norm_epsilon": 1e-6,
        "rope_theta": 10000.0,
        "vocab_size": 2,
        "reg_token_id": 1,
    }
    cfg = Chronos2MLXConfig.from_hf_config(hf)
    assert cfg.d_model == 768
    assert cfg.num_quantiles == 3
    assert cfg.use_arcsinh is True
