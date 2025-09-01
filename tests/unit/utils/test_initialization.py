"""Tests for weight initialization functions via SOM.initialize_weights."""

import pytest
import torch

from torchsom.core.som import SOM

pytestmark = [
    pytest.mark.unit,
]


class TestWeightInitialization:
    def test_initialize_weights_random(
        self,
        som_small: SOM,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test random weight initialization."""
        data = small_random_data.to(som_small.device)
        original_weights = som_small.weights.clone()
        som_small.initialize_weights(data, mode="random")

        # Weights should have changed
        assert not torch.allclose(original_weights, som_small.weights)

        # Weights should be within the data range (min/max) for each feature
        data_min = data.min(dim=0).values
        data_max = data.max(dim=0).values

        # Check that all weights are within the min/max of the data for each feature
        assert torch.all(som_small.weights >= data_min - 1e-6)
        assert torch.all(som_small.weights <= data_max + 1e-6)

        # Optionally, check for NaNs or infs
        assert not torch.isnan(som_small.weights).any()
        assert not torch.isinf(som_small.weights).any()

    def test_initialize_weights_pca(
        self,
        som_small: SOM,
        medium_random_data: torch.Tensor,
    ) -> None:
        """Test PCA weight initialization."""
        data = medium_random_data.to(som_small.device)
        original_weights = som_small.weights.clone()
        som_small.initialize_weights(data, mode="pca")

        # Weights should have changed
        assert not torch.allclose(original_weights, som_small.weights)

        # Weights should be within the data range (min/max) for each feature
        data_min = data.min(dim=0).values
        data_max = data.max(dim=0).values

        # Check that all weights are within the min/max of the data for each feature
        assert torch.all(som_small.weights >= data_min - 1e-6)
        assert torch.all(som_small.weights <= data_max + 1e-6)

        # Optionally, check for NaNs or infs
        assert not torch.isnan(som_small.weights).any()
        assert not torch.isinf(som_small.weights).any()

    def test_initialize_weights_invalid_mode(
        self,
        som_small: SOM,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test that invalid initialization mode raises error."""
        data = small_random_data.to(som_small.device)
        with pytest.raises(ValueError):
            som_small.initialize_weights(data, mode="invalid_mode")
