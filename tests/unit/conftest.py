"""Consolidated plugin loader for pytest fixtures.

This conftest delegates fixtures to tests/fixtures/ modules via pytest_plugins.
"""

pytest_plugins = [
    "tests.fixtures.data",
    "tests.fixtures.devices",
    "tests.fixtures.markers",
    "tests.fixtures.som_configs",
    "tests.fixtures.som_instances",
    "tests.fixtures.som_parameters",
]
