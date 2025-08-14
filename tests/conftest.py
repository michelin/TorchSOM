"""Shared fixtures and configuration for TorchSOM tests."""

from typing import Any

import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs, make_regression

from torchsom.core.som import SOM

# ==================== SOM Initialization Fixtures ====================


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> str:
    """Parametrized device fixture for testing CPU/GPU compatibility."""
    device_name = request.param
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return device_name


@pytest.fixture
def fixed_seed() -> int:
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture(params=["rectangular", "hexagonal"])
def topology(request) -> str:
    """Parametrized topology fixture for testing both topologies."""
    return request.param


@pytest.fixture(params=["euclidean", "cosine", "manhattan", "chebyshev"])
def distance_function(request) -> str:
    """Parametrized distance function fixture."""
    return request.param


@pytest.fixture(params=["gaussian", "bubble", "triangle", "mexican_hat"])
def neighborhood_function(request) -> str:
    """Parametrized neighborhood function fixture."""
    return request.param


@pytest.fixture(
    params=["lr_inverse_decay_to_zero", "lr_linear_decay_to_zero", "asymptotic_decay"]
)
def lr_decay_function(request) -> str:
    """Parametrized learning rate decay function fixture."""
    return request.param


@pytest.fixture(
    params=["sig_inverse_decay_to_one", "sig_linear_decay_to_one", "asymptotic_decay"]
)
def sigma_decay_function(request) -> str:
    """Parametrized sigma decay function fixture."""
    return request.param


@pytest.fixture(params=["random", "pca"])
def initialization_mode(request) -> str:
    """Parametrized initialization mode fixture."""
    return request.param


@pytest.fixture(params=["mean", "std"])
def reduction_parameter(request) -> str:
    """Parametrized reduction parameter fixture."""
    return request.param


@pytest.fixture(params=[True, False])
def return_indices(request) -> bool:
    """Parametrized return indices fixture."""
    return request.param


# ==================== Device and Reproducibility Fixtures ====================


# ! autouse=True means that the fixture will be applied automatically to test functions without including it as an argument
@pytest.fixture(autouse=True)
def set_random_seeds(fixed_seed) -> None:
    """Set all random seeds for reproducibility across the test suite."""
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fixed_seed)
        torch.cuda.manual_seed_all(fixed_seed)


# ==================== Data Generation Fixtures ====================


@pytest.fixture
def single_sample() -> torch.Tensor:
    """Single data sample for BMU identification tests.

    Returns:
        torch.Tensor: 1D tensor of shape [4] to match standard SOM configuration
    """
    return torch.randn(4)


@pytest.fixture
def small_random_data() -> torch.Tensor:
    """Small random standardized dataset for quick unit tests.

    Returns:
        torch.Tensor: 2D tensor of shape [50, 4] with normalized random data
    """
    data = torch.randn(50, 4)
    return (data - data.mean(dim=0)) / data.std(dim=0)


@pytest.fixture
def medium_random_data() -> torch.Tensor:
    """Medium-sized random standardized dataset for integration tests.

    Returns:
        torch.Tensor: 2D tensor of shape [2000, 4] with normalized random data
    """
    data = torch.randn(2000, 4)
    return (data - data.mean(dim=0)) / data.std(dim=0)


@pytest.fixture
def high_dim_data() -> torch.Tensor:
    """High-dimensional standardized data for testing scalability.

    Returns:
        torch.Tensor: 2D tensor of shape [1000, 100] with normalized random data
    """
    data = torch.randn(1000, 100)
    return (data - data.mean(dim=0)) / data.std(dim=0)


@pytest.fixture
def clustered_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate clustered standardized data using sklearn make_blobs for topology validation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (data, labels) where data is [200, 4] and labels [200]
    """
    X, y = make_blobs(
        n_samples=200, centers=3, n_features=4, cluster_std=1.0, random_state=42
    )
    data = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.long)

    data = (data - data.mean(dim=0)) / data.std(dim=0)
    return data, labels


@pytest.fixture
def regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate regression data for testing."""
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=42)
    data = torch.tensor(X, dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    return data, target


# ==================== SOM Configuration Fixtures ====================


@pytest.fixture
def som_config_minimal(
    device: str,
    topology: str,
) -> dict[str, Any]:
    """Minimal SOM configuration for basic tests."""
    return {
        "x": 5,
        "y": 5,
        "topology": topology,
        "num_features": 4,
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 0.9,
        "sigma": 1.25,
        "neighborhood_function": "gaussian",
        "distance_function": "euclidean",
        "initialization_mode": "random",
        "lr_decay_function": "asymptotic_decay",
        "sigma_decay_function": "asymptotic_decay",
        "neighborhood_order": 1,
        "random_seed": 42,
        "device": device,
    }


@pytest.fixture
def som_config_standard(
    device: str,
    topology: str,
) -> dict[str, Any]:
    """Standard SOM configuration for comprehensive tests."""
    return {
        "x": 15,
        "y": 10,
        "topology": topology,
        "num_features": 4,
        "epochs": 15,
        "batch_size": 32,
        "sigma": 2.0,
        "learning_rate": 0.85,
        "neighborhood_function": "gaussian",
        "distance_function": "euclidean",
        "initialization_mode": "random",
        "lr_decay_function": "asymptotic_decay",
        "sigma_decay_function": "asymptotic_decay",
        "neighborhood_order": 2,
        "random_seed": 42,
        "device": device,
    }


@pytest.fixture
def som_config_comprehensive(
    topology: str,
    distance_function: str,
    neighborhood_function: str,
    lr_decay_function: str,
    sigma_decay_function: str,
    initialization_mode: str,
    fixed_seed: int,
    device: str,
) -> dict[str, Any]:
    """Comprehensive SOM configuration testing multiple parameter combinations."""
    return {
        "x": 8,
        "y": 6,
        "num_features": 4,
        "epochs": 10,
        "batch_size": 16,
        "sigma": 1.10,
        "learning_rate": 0.85,
        "topology": topology,
        "neighborhood_function": neighborhood_function,
        "distance_function": distance_function,
        "initialization_mode": initialization_mode,
        "lr_decay_function": lr_decay_function,
        "sigma_decay_function": sigma_decay_function,
        "neighborhood_order": 2,
        "random_seed": fixed_seed,
        "device": device,
    }


# ==================== SOM Instance Fixtures ====================


@pytest.fixture
def som_small(
    som_config_minimal: dict[str, Any],
) -> SOM:
    """Small SOM instance for quick unit tests.

    Args:
        som_config_minimal: Minimal configuration

    Returns:
        SOM: Initialized minimal SOM instance
    """
    return SOM(**som_config_minimal)


@pytest.fixture
def som_standard(
    som_config_standard: dict[str, Any],
) -> SOM:
    """Standard SOM instance for comprehensive tests.

    Args:
        som_config_standard: Standard configuration

    Returns:
        SOM: Initialized standard SOM instance
    """
    return SOM(**som_config_standard)


@pytest.fixture
def som_comprehensive(
    som_config_comprehensive: dict[str, Any],
) -> SOM:
    """Comprehensive SOM instance for comprehensive tests.

    Args:
        som_config_comprehensive: Comprehensive configuration

    Returns:
        SOM: Initialized comprehensive SOM instance
    """
    return SOM(**som_config_comprehensive)


@pytest.fixture
def som_trained(
    som_standard: SOM,
    medium_random_data: torch.Tensor,
) -> SOM:
    """Pre-trained SOM for testing post-training functionality.

    Args:
        som_standard: Standard SOM instance
        medium_random_data: Training data

    Returns:
        SOM: Trained SOM instance
    """
    data = medium_random_data.to(som_standard.device)
    som_standard.fit(data)
    return som_standard


# # ==================== Configuration Object Fixtures ====================


# @pytest.fixture
# def som_config_object() -> SOMConfig:
#     """SOMConfig pydantic object for configuration validation tests."""
#     return SOMConfig(
#         x=8,
#         y=8,
#         epochs=15,
#         batch_size=8,
#         learning_rate=0.4,
#         sigma=1.5,
#         topology="rectangular",
#         neighborhood_function="gaussian",
#         distance_function="euclidean",
#     )


# # ==================== Edge Case Data Fixtures ====================


# @pytest.fixture
# def edge_case_data() -> Dict[str, torch.Tensor]:
#     """Various edge case datasets for robustness testing."""
#     return {
#         "empty": torch.empty(0, 4),
#         "single_point": torch.randn(1, 4),
#         "zeros": torch.zeros(10, 4),
#         "ones": torch.ones(10, 4),
#         "large_values": torch.randn(10, 4) * 1000,
#         "small_values": torch.randn(10, 4) * 1e-6,
#         "inf_values": torch.full((5, 4), float("inf")),
#         "nan_values": torch.full((5, 4), float("nan")),
#     }


# # ==================== Test Skip Conditions ====================


# def pytest_collection_modifyitems(
#     config,
#     items,
# ):
#     """Modify test collection to add skip conditions."""
#     for item in items:
#         # Skip GPU tests if CUDA unavailable
#         if "gpu" in item.keywords and not torch.cuda.is_available():
#             item.add_marker(pytest.mark.skip(reason="CUDA not available"))


# ==================== Test Markers Configuration ====================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit test markers")
    config.addinivalue_line("markers", "integration: Integration test markers")
    config.addinivalue_line("markers", "slow: Tests that take more than a few seconds")
    config.addinivalue_line("markers", "gpu: Tests that require CUDA")
    config.addinivalue_line(
        "markers", "mathematical: Tests validating mathematical correctness"
    )
