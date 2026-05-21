"""SOM config fixtures."""

from typing import Any

import pytest


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
