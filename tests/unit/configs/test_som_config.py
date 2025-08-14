"""Unit tests for `torchsom.configs.som_config.SOMConfig`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from torchsom.configs.som_config import SOMConfig

# Mark all tests in this module as unit tests
pytestmark = [
    pytest.mark.unit,
    # pytest.mark.gpu,
]


def test_som_config_comprehensive_defaults(
    som_config_comprehensive: dict[str, Any],
) -> None:
    """SOMConfig builds with only required fields and applies sane defaults."""
    cfg = SOMConfig(**som_config_comprehensive)

    assert cfg.x == som_config_comprehensive["x"]
    assert cfg.y == som_config_comprehensive["y"]
    assert cfg.topology == som_config_comprehensive["topology"]
    assert cfg.epochs == som_config_comprehensive["epochs"]
    assert cfg.batch_size == som_config_comprehensive["batch_size"]
    assert cfg.learning_rate == som_config_comprehensive["learning_rate"]
    assert cfg.sigma == som_config_comprehensive["sigma"]
    assert (
        cfg.neighborhood_function == som_config_comprehensive["neighborhood_function"]
    )
    assert cfg.distance_function == som_config_comprehensive["distance_function"]
    assert cfg.lr_decay_function == som_config_comprehensive["lr_decay_function"]
    assert cfg.sigma_decay_function == som_config_comprehensive["sigma_decay_function"]
    assert cfg.initialization_mode == som_config_comprehensive["initialization_mode"]
    assert cfg.neighborhood_order == som_config_comprehensive["neighborhood_order"]
    assert cfg.random_seed == som_config_comprehensive["random_seed"]
    assert cfg.device == som_config_comprehensive["device"]


def test_som_config_repr_contains_key_fields() -> None:
    """String repr should include core fields for debugging convenience."""
    cfg = SOMConfig(x=3, y=7)
    text = repr(cfg)
    assert "x=3" in text and "y=7" in text


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"x": 0, "y": 5},  # x must be > 0
        {"x": 5, "y": 0},  # y must be > 0
        {"x": 5, "y": 5, "epochs": 0},  # epochs >= 1
        {"x": 5, "y": 5, "batch_size": 0},  # batch_size >= 1
        {"x": 5, "y": 5, "learning_rate": 0.0},  # lr > 0
        {"x": 5, "y": 5, "sigma": 0.0},  # sigma > 0
        {"x": 5, "y": 5, "neighborhood_order": 0},  # order >= 1
        {"x": 5, "y": 5, "topology": "triangle"},  # invalid enum
        {"x": 5, "y": 5, "distance_function": "minkowski"},  # invalid enum
        {"x": 5, "y": 5, "neighborhood_function": "lorentzian"},  # invalid enum
        {"x": 5, "y": 5, "lr_decay_function": "linear"},  # invalid enum
        {"x": 5, "y": 5, "sigma_decay_function": "linear"},  # invalid enum
        {"x": 5, "y": 5, "initialization_mode": "kmeans"},  # invalid enum
    ],
)
def test_som_config_validation_errors(
    bad_kwargs: dict,
) -> None:
    """Invalid values should raise pydantic ValidationError."""
    with pytest.raises(ValidationError):
        SOMConfig(**bad_kwargs)
