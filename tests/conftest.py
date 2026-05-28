"""Shared fixtures for parity tests."""
import numpy as np
import pytest
import torch
import mlx.core as mx

from chronos2_mlx import Chronos2MLXPipeline


@pytest.fixture(scope="session")
def torch_model():
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        attn_implementation="eager",
    )
    pipe.model.eval()
    return pipe.model


@pytest.fixture(scope="session")
def mlx_pipeline():
    return Chronos2MLXPipeline.from_pretrained("amazon/chronos-2")


@pytest.fixture(scope="session")
def mlx_model(mlx_pipeline):
    return mlx_pipeline.model


def check_close(mlx_arr, torch_arr, atol=1e-4, name=""):
    m = np.array(mlx_arr)
    t = torch_arr.float().detach().numpy()
    assert np.allclose(m, t, atol=atol), (
        f"{name}: max_ae={np.abs(m-t).max():.2e}, mean_ae={np.abs(m-t).mean():.2e}"
    )
