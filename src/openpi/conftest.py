import os
import warnings

import pytest

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The pynvml package is deprecated.*",
        category=FutureWarning,
    )
    import pynvml


def set_jax_cpu_backend_if_no_gpu() -> None:
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # No GPU found.
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()
