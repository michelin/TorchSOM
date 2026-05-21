"""Data generation fixtures for tests."""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs, make_regression


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


@pytest.fixture
def well_separated_clusters() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate well-separated clusters for clustering algorithm testing."""
    X, y = make_blobs(
        n_samples=300,
        centers=4,
        n_features=3,
        cluster_std=0.5,
        center_box=(-10.0, 10.0),
        random_state=42,
    )
    data = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.long)
    data = (data - data.mean(dim=0)) / data.std(dim=0)
    return data, labels


@pytest.fixture
def noisy_clustering_data() -> torch.Tensor:
    """Generate noisy data for testing clustering robustness."""
    X, _ = make_blobs(
        n_samples=100,
        centers=3,
        n_features=2,
        cluster_std=0.8,
        random_state=42,
    )
    noise = np.random.RandomState(42).uniform(-5, 5, size=(50, 2))
    X_with_noise = np.vstack([X, noise])
    data = torch.tensor(X_with_noise, dtype=torch.float32)
    data = (data - data.mean(dim=0)) / data.std(dim=0)
    return data


@pytest.fixture
def hexagonal_test_coordinates() -> list[tuple[int, int]]:
    """Generate coordinates for hexagonal grid testing."""
    return [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
    ]
