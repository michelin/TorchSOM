"""Pytest markers configuration plugin."""

import pytest


def pytest_configure(
    config: pytest.Config,
) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit test markers")
    config.addinivalue_line("markers", "integration: Integration test markers")
    config.addinivalue_line("markers", "slow: Tests that take more than a few seconds")
    config.addinivalue_line("markers", "gpu: Tests that require CUDA")
    config.addinivalue_line(
        "markers", "mathematical: Tests validating mathematical correctness"
    )
