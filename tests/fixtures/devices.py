"""Device and reproducibility fixtures."""

import numpy as np
import pytest
import torch


@pytest.fixture
def fixed_seed() -> int:
    """Fixed seed for reproducibility."""
    return 42


@pytest.fixture(
    params=[
        "cpu",
        "cuda",
    ]
)
def device(request: pytest.FixtureRequest) -> str:
    """Device for tensor computations."""
    device_name = request.param
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return device_name


# ! autouse=True means that the fixture will be applied automatically to test functions without including it as an argument
@pytest.fixture(autouse=True)
def set_random_seeds(fixed_seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fixed_seed)
        torch.cuda.manual_seed_all(fixed_seed)
