"""Tests for clustering utilities in torchsom.utils.clustering."""

from typing import Any

import numpy as np
import pytest
import torch

from torchsom.core.som import SOM
from torchsom.utils.clustering import (
    _determine_optimal_components_bic,
    _determine_optimal_k_elbow,
    cluster_data,
    cluster_gmm,
    cluster_kmeans,
    extract_clustering_features,
)

pytestmark = [
    pytest.mark.unit,
]


class TestSOMClustering:
    def test_som_cluster_basic(
        self,
        som_trained: SOM,
        clustering_method: str,
    ) -> None:
        """Test basic SOM clustering functionality."""
        result = som_trained.cluster(method=clustering_method, n_clusters=3)
        assert isinstance(result, dict)
        assert "labels" in result
        assert "centers" in result
        assert "method" in result
        assert result["method"] == clustering_method
        assert result["labels"].shape == (som_trained.x * som_trained.y,)
        assert result["labels"].device.type == som_trained.device

    def test_som_cluster_auto_clusters(
        self,
        som_trained: SOM,
        clustering_method: str,
    ) -> None:
        """Test SOM clustering with automatic cluster number selection."""
        result = som_trained.cluster(method=clustering_method, n_clusters=None)
        assert isinstance(result, dict)
        assert result["n_clusters"] >= 1
        assert result["labels"].max() <= result["n_clusters"]

    def test_som_cluster_feature_space_weights(
        self,
        som_trained: SOM,
    ) -> None:
        """Test clustering using neuron weights as features."""
        result = som_trained.cluster(
            method="kmeans", n_clusters=3, feature_space="weights"
        )
        assert isinstance(result, dict)
        assert result["labels"].shape == (som_trained.x * som_trained.y,)

    def test_clustering_device_consistency(
        self,
        som_config_minimal: dict[str, Any],
        small_random_data: torch.Tensor,
        clustering_method: str,
    ) -> None:
        """Test clustering consistency across devices."""
        # Test CPU
        som_cpu = SOM(**{**som_config_minimal, "device": "cpu"})
        data_cpu = small_random_data.to("cpu")
        som_cpu.fit(data_cpu)
        result_cpu = som_cpu.cluster(
            method=clustering_method, n_clusters=2, random_state=42
        )
        if torch.cuda.is_available():
            # Test GPU
            som_gpu = SOM(**{**som_config_minimal, "device": "cuda"})
            data_gpu = small_random_data.to("cuda")
            som_gpu.fit(data_gpu)
            result_gpu = som_gpu.cluster(
                method=clustering_method, n_clusters=2, random_state=42
            )
            assert result_cpu["labels"].shape == result_gpu["labels"].cpu().shape
            assert result_cpu["n_clusters"] == result_gpu["n_clusters"]

    def test_clustering_reproducibility(
        self,
        som_trained: SOM,
        clustering_method: str,
    ) -> None:
        """Test that clustering results are reproducible."""
        result1 = som_trained.cluster(
            method=clustering_method, n_clusters=3, random_state=42
        )
        result2 = som_trained.cluster(
            method=clustering_method, n_clusters=3, random_state=42
        )
        torch.testing.assert_close(result1["labels"], result2["labels"])
        torch.testing.assert_close(result1["centers"], result2["centers"])

    def test_clustering_parameter_passing(
        self,
        som_trained: SOM,
    ) -> None:
        """Test that clustering parameters are passed correctly."""
        # Test K-means with specific parameters
        result = som_trained.cluster(
            method="kmeans",
            n_clusters=2,
            random_state=42,
            max_iter=50,  # Additional K-means parameter
        )
        assert result["n_clusters"] == 2

        # # Test HDBSCAN with specific parameters
        # result = som_trained.cluster(
        #     method="hdbscan",
        #     min_cluster_size=5,  # HDBSCAN parameter
        # )

        # assert isinstance(result["noise_points"], int)

    def test_extract_clustering_features(
        self,
        som_trained: SOM,
        clustering_space: str,
    ) -> None:
        """Test feature extraction for clustering."""
        features = extract_clustering_features(
            som=som_trained, feature_space=clustering_space
        )
        if clustering_space == "weights":
            expected_shape = (som_trained.x * som_trained.y, som_trained.num_features)
        elif clustering_space == "positions":
            expected_shape = (som_trained.x * som_trained.y, 2)
        elif clustering_space == "combined":
            expected_shape = (
                som_trained.x * som_trained.y,
                som_trained.num_features + 2,
            )
        assert features.shape == expected_shape
        assert features.device.type == som_trained.device
        # Verify features are normalized weights
        expected_features = som_trained.weights.view(-1, som_trained.num_features)
        if clustering_space == "weights":
            torch.testing.assert_close(features, expected_features)

    def test_build_classification_map_edge_cases(
        self,
        som_small: SOM,
    ) -> None:
        """Test classification map building edge cases."""
        data = torch.randn(10, som_small.num_features, device=som_small.device)
        labels = torch.randint(0, 3, (10,), device=som_small.device)
        classification_map = som_small.build_map(
            "classification", data=data, target=labels, neighborhood_order=0
        )
        assert torch.isnan(classification_map).any()

    def test_clustering_small_som(
        self,
        clustering_method: str,
        device: str,
    ) -> None:
        """Test clustering on very small SOM."""
        som = SOM(x=2, y=2, num_features=3, epochs=5, device=device, random_seed=42)
        data = torch.randn(20, 3, device=device)
        som.fit(data)

        result = som.cluster(method=clustering_method, n_clusters=2)

        assert result["labels"].shape == (4,)  # 2x2 = 4 neurons
        assert result["n_clusters"] >= 1


class TestClusteringUtilities:
    def test_cluster_kmeans_basic(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
        device: str,
    ) -> None:
        """Test basic K-means clustering functionality."""
        data, expected_labels = well_separated_clusters
        data = data.to(device)

        # Test with known number of clusters
        result = cluster_kmeans(data, n_clusters=4, random_state=42)

        assert isinstance(result, dict)
        assert "labels" in result
        assert "centers" in result
        assert "n_clusters" in result
        assert "method" in result
        assert "inertia" in result

        assert result["labels"].shape == (data.shape[0],)
        assert result["centers"].shape == (4, data.shape[1])
        assert result["n_clusters"] == 4
        assert result["method"] == "kmeans"
        assert result["labels"].device == data.device
        assert result["centers"].device == data.device
        assert torch.all(result["labels"] >= 1)  # 1-indexed labels

    def test_cluster_kmeans_auto_k(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
        device: str,
    ) -> None:
        """Test K-means with automatic k selection."""
        data, _ = well_separated_clusters
        data = data.to(device)

        # Test with automatic k selection
        result = cluster_kmeans(data, n_clusters=None, random_state=42)

        assert isinstance(result, dict)
        assert result["n_clusters"] >= 2
        assert result["labels"].max() == result["n_clusters"]
        assert result["centers"].shape[0] == result["n_clusters"]

    def test_cluster_gmm_basic(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
        device: str,
    ) -> None:
        """Test basic GMM clustering functionality."""
        data, _ = well_separated_clusters
        data = data.to(device)

        result = cluster_gmm(data, n_components=3, random_state=42)

        assert isinstance(result, dict)
        assert "labels" in result
        assert "centers" in result
        assert "n_clusters" in result
        assert "method" in result
        assert "bic" in result
        assert "aic" in result

        assert result["labels"].shape == (data.shape[0],)
        assert result["centers"].shape == (3, data.shape[1])
        assert result["n_clusters"] == 3
        assert result["method"] == "gmm"
        assert result["labels"].device == data.device
        assert result["centers"].device == data.device

    def test_cluster_gmm_auto_components(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
        device: str,
    ) -> None:
        """Test GMM with automatic component selection."""
        data, _ = well_separated_clusters
        data = data.to(device)

        result = cluster_gmm(data, n_components=None, random_state=42)

        assert isinstance(result, dict)
        assert result["n_clusters"] >= 1
        assert isinstance(result["bic"], (float, np.floating))
        assert isinstance(result["aic"], (float, np.floating))

    def test_cluster_data_dispatcher(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
        clustering_method: str,
        device: str,
    ) -> None:
        """Test main cluster_data dispatcher function."""
        data, _ = well_separated_clusters
        data = data.to(device)

        result = cluster_data(data, method=clustering_method, n_clusters=3)

        assert isinstance(result, dict)
        assert "labels" in result
        assert "centers" in result
        assert "method" in result
        assert result["method"] == clustering_method
        assert result["labels"].device == data.device

    def test_cluster_data_invalid_method(
        self,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test cluster_data with invalid method."""
        with pytest.raises(ValueError, match="Unsupported clustering method"):
            cluster_data(small_random_data, method="invalid_method")

    def test_cluster_data_input_validation(
        self,
    ) -> None:
        """Test cluster_data input validation."""
        with pytest.raises(ValueError, match="Cannot cluster empty data"):
            cluster_data(torch.empty(0, 4))
        with pytest.raises(ValueError, match="Data must be 2D tensor"):
            cluster_data(torch.randn(10))
        with pytest.raises(ValueError, match="Need at least 2 samples"):
            cluster_data(torch.randn(1, 4))

    def test_determine_optimal_k_elbow(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test elbow method for optimal k determination."""
        data, _ = well_separated_clusters
        data_np = data.detach().cpu().numpy()

        optimal_k = _determine_optimal_k_elbow(data_np, max_k=8, random_state=42)

        assert isinstance(optimal_k, int)
        assert 2 <= optimal_k <= 8

    def test_determine_optimal_k_edge_cases(
        self,
    ) -> None:
        """Test elbow method edge cases."""
        # Very small dataset
        small_data = np.random.randn(3, 2)
        optimal_k = _determine_optimal_k_elbow(small_data, random_state=42)
        assert optimal_k == 2  # Should return minimum

    def test_determine_optimal_components_bic(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test BIC method for optimal component determination."""
        data, _ = well_separated_clusters
        data_np = data.detach().cpu().numpy()

        optimal_components = _determine_optimal_components_bic(
            data_np, max_components=6, random_state=42
        )

        assert isinstance(optimal_components, int)
        assert 1 <= optimal_components <= 6

    def test_clustering_device_consistency(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
        clustering_method: str,
    ) -> None:
        """Test that clustering results are consistent across devices."""
        data, _ = well_separated_clusters

        # Test on CPU
        result_cpu = cluster_data(
            data.to("cpu"), method=clustering_method, n_clusters=3, random_state=42
        )

        if torch.cuda.is_available():
            # Test on GPU
            result_gpu = cluster_data(
                data.to("cuda"), method=clustering_method, n_clusters=3, random_state=42
            )

            # Results should be equivalent (accounting for device)
            assert result_cpu["labels"].shape == result_gpu["labels"].cpu().shape
            assert result_cpu["centers"].shape == result_gpu["centers"].cpu().shape

    def test_clustering_small_datasets(
        self,
        clustering_method: str,
    ) -> None:
        """Test clustering with very small datasets."""
        # Create minimal viable dataset
        small_data = torch.randn(5, 3)

        result = cluster_data(small_data, method=clustering_method, n_clusters=2)

        assert isinstance(result, dict)
        assert result["labels"].shape == (5,)
        assert result["n_clusters"] >= 1

    # def test_cluster_hdbscan_basic(
    #     self,
    #     noisy_clustering_data: torch.Tensor,
    #     device: str,
    # ) -> None:
    #     """Test basic HDBSCAN clustering functionality."""
    #     data = noisy_clustering_data.to(device)

    #     result = cluster_hdbscan(data, min_cluster_size=10)

    #     assert isinstance(result, dict)
    #     assert "labels" in result
    #     assert "centers" in result
    #     assert "n_clusters" in result
    #     assert "method" in result
    #     assert "noise_points" in result

    #     assert result["labels"].shape == (data.shape[0],)
    #     assert result["method"] == "hdbscan"
    #     assert result["labels"].device == data.device
    #     assert result["centers"].device == data.device
    #     assert isinstance(result["noise_points"], int)
    #     assert result["noise_points"] >= 0

    # def test_cluster_hdbscan_auto_min_size(
    #     self,
    #     well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
    #     device: str,
    # ) -> None:
    #     """Test HDBSCAN with automatic min_cluster_size."""
    #     data, _ = well_separated_clusters
    #     data = data.to(device)

    #     result = cluster_hdbscan(data, min_cluster_size=None)

    #     assert isinstance(result, dict)
    #     assert result["n_clusters"] >= 0  # Could be 0 if all noise
