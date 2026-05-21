"""SOM instance fixtures and configurations composed from other fixtures."""

from typing import Any

import pytest
import torch

from torchsom.core.som import SOM


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
