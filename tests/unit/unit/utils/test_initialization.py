"""Tests for weight initialization functions via SOM.initialize_weights."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

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

    def test_initialize_weights_random_dataloader(
        self,
        som_small: SOM,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test random weight initialization via DataLoader (reservoir sampling)."""
        loader = DataLoader(TensorDataset(small_random_data), batch_size=16, shuffle=False)
        original_weights = som_small.weights.clone()
        som_small.initialize_weights(loader, mode="random")

        assert som_small.weights.shape == original_weights.shape
        assert not torch.allclose(original_weights, som_small.weights)
        assert not torch.isnan(som_small.weights).any()
        assert not torch.isinf(som_small.weights).any()

    def test_initialize_weights_pca_dataloader(
        self,
        som_small: SOM,
        medium_random_data: torch.Tensor,
    ) -> None:
        """Test PCA weight initialization via DataLoader (Chan's algorithm)."""
        loader = DataLoader(TensorDataset(medium_random_data), batch_size=64, shuffle=False)
        original_weights = som_small.weights.clone()
        som_small.initialize_weights(loader, mode="pca")

        assert som_small.weights.shape == original_weights.shape
        assert not torch.allclose(original_weights, som_small.weights)
        assert not torch.isnan(som_small.weights).any()
        assert not torch.isinf(som_small.weights).any()
