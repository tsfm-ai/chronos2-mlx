from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class Chronos2MLXConfig:
    d_model: int = 768
    d_kv: int = 64
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    context_length: int = 8192
    input_patch_size: int = 16
    input_patch_stride: int = 16
    output_patch_size: int = 16
    max_output_patches: int = 64
    quantiles: Sequence[float] = field(default_factory=lambda: [
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
    ])
    use_arcsinh: bool = True
    use_reg_token: bool = True
    reg_token_id: int = 1
    rope_theta: float = 10000.0
    layer_norm_epsilon: float = 1e-6
    time_encoding_scale: float = 8192.0
    vocab_size: int = 4096
    max_covariates: int = 512

    @classmethod
    def from_hf_config(cls, hf_config: dict) -> "Chronos2MLXConfig":
        cc = hf_config.get("chronos_config", hf_config)
        return cls(
            d_model=cc.get("d_model", 768),
            d_kv=cc.get("d_kv", 64),
            d_ff=cc.get("d_ff", 3072),
            num_layers=cc.get("num_layers", 12),
            num_heads=cc.get("num_heads", 12),
            context_length=cc.get("context_length", 8192),
            input_patch_size=cc.get("input_patch_size", 16),
            input_patch_stride=cc.get("input_patch_stride", 16),
            output_patch_size=cc.get("output_patch_size", 16),
            max_output_patches=cc.get("max_output_patches", 64),
            quantiles=cc.get("quantiles", cls.__dataclass_fields__["quantiles"].default_factory()),
            use_arcsinh=cc.get("use_arcsinh", True),
            use_reg_token=cc.get("use_reg_token", True),
            reg_token_id=cc.get("reg_token_id", 1),
            rope_theta=cc.get("rope_theta", 10000.0),
            layer_norm_epsilon=cc.get("layer_norm_epsilon", 1e-6),
            time_encoding_scale=cc.get("time_encoding_scale", 8192.0),
            vocab_size=cc.get("vocab_size", 4096),
            max_covariates=cc.get("max_covariates", 512),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Chronos2MLXConfig":
        config = json.loads(Path(path).read_text())
        return cls.from_hf_config(config)

    @property
    def num_quantiles(self) -> int:
        return len(self.quantiles)

    @property
    def num_context_patches(self) -> int:
        return 1 + (self.context_length - self.input_patch_size) // self.input_patch_stride

    @property
    def d_patch_input(self) -> int:
        return self.input_patch_size

    @property
    def d_patch_output(self) -> int:
        return self.output_patch_size * self.num_quantiles
