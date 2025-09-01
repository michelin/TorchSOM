"""Tests for clustering quality metrics in torchsom.utils.metrics."""

import pytest
import torch

from torchsom.core.som import SOM
from torchsom.utils.metrics import (
    calculate_calinski_harabasz_score,
    calculate_clustering_metrics,
    calculate_davies_bouldin_score,
    calculate_silhouette_score,
    calculate_topological_clustering_quality,
)

pytestmark = [
    pytest.mark.unit,
]


class TestClusteringMetrics:
    def test_calculate_silhouette_score_valid_clusters(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
        device: str,
    ) -> None:
        """Test that the silhouette score is calculated correctly."""
        data, labels = well_separated_clusters
        data = data.to(device)
        labels = labels.to(device)
        score = calculate_silhouette_score(data, labels)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
        assert score > 0.5

    def test_calculate_silhouette_score_with_noise(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
        device: str,
    ) -> None:
        """Test that the silhouette score is calculated correctly with noise."""
        data, labels = well_separated_clusters
        data = data.to(device)
        labels = labels.to(device)
        labels[:10] = -1  # Intentional noise
        score = calculate_silhouette_score(data, labels)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_calculate_davies_bouldin_score(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
        device: str,
    ) -> None:
        """Test that the Davies-Bouldin score is calculated correctly."""
        data, labels = well_separated_clusters
        data = data.to(device)
        labels = labels.to(device)
        score = calculate_davies_bouldin_score(data, labels)
        assert isinstance(score, float)
        assert score >= 0.0
        assert score < 2.0

    def test_calculate_calinski_harabasz_score(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
        device: str,
    ) -> None:
        """Test that the Calinski-Harabasz score is calculated correctly."""
        data, labels = well_separated_clusters
        data = data.to(device)
        labels = labels.to(device)
        score = calculate_calinski_harabasz_score(data, labels)
        assert isinstance(score, float)
        assert score >= 0.0
        assert score > 10.0

    def test_metrics_with_single_cluster(
        self,
        small_random_data: torch.Tensor,
        device: str,
    ) -> None:
        """Test that the metrics are calculated correctly with a single cluster."""
        data = small_random_data.to(device)
        labels = torch.ones(data.shape[0], dtype=torch.long, device=device)
        silhouette = calculate_silhouette_score(data, labels)
        davies_bouldin = calculate_davies_bouldin_score(data, labels)
        calinski_harabasz = calculate_calinski_harabasz_score(data, labels)
        assert silhouette == 0.0
        assert davies_bouldin == 0.0
        assert calinski_harabasz == 0.0

    def test_metrics_with_all_noise(
        self,
        small_random_data: torch.Tensor,
        device: str,
    ) -> None:
        """Test that the metrics are calculated correctly with all noise."""
        data = small_random_data.to(device)
        labels = torch.full((data.shape[0],), -1, dtype=torch.long, device=device)
        silhouette = calculate_silhouette_score(data, labels)
        davies_bouldin = calculate_davies_bouldin_score(data, labels)
        calinski_harabasz = calculate_calinski_harabasz_score(data, labels)
        assert silhouette == 0.0
        assert davies_bouldin == float("inf")
        assert calinski_harabasz == 0.0

    def test_calculate_topological_clustering_quality_rectangular(
        self,
        som_trained: SOM,
    ) -> None:
        """Test that the topological clustering quality is calculated correctly for a rectangular SOM."""
        labels = torch.zeros(som_trained.x * som_trained.y, dtype=torch.long)
        labels[: som_trained.x * som_trained.y // 2] = 1
        labels[som_trained.x * som_trained.y // 2 :] = 2
        quality = calculate_topological_clustering_quality(som_trained, labels)
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0

    def test_calculate_topological_clustering_quality_hexagonal(
        self,
        som_config_minimal: dict[str, object],
        device: str,
    ) -> None:
        """Test that the topological clustering quality is calculated correctly for a hexagonal SOM."""
        cfg = {**som_config_minimal, "topology": "hexagonal", "device": device}
        som = SOM(**cfg)
        labels = torch.zeros(som.x * som.y, dtype=torch.long)
        labels[: som.x * som.y // 2] = 1
        labels[som.x * som.y // 2 :] = 2
        quality = calculate_topological_clustering_quality(som, labels)
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0

    def test_topological_quality_single_cluster(
        self,
        som_small: SOM,
    ) -> None:
        """Test that the topological clustering quality is calculated correctly for a single cluster."""
        labels = torch.ones(som_small.x * som_small.y, dtype=torch.long)
        quality = calculate_topological_clustering_quality(som_small, labels)
        assert quality == 1.0  # Single cluster should be perfectly topological

    def test_topological_quality_with_noise(
        self,
        som_small: SOM,
    ) -> None:
        """Test that the topological clustering quality is calculated correctly with noise."""
        labels = torch.ones(som_small.x * som_small.y, dtype=torch.long)
        labels[:5] = -1  # Intentional noise
        quality = calculate_topological_clustering_quality(som_small, labels)
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0

    def test_calculate_clustering_metrics_comprehensive(
        self,
        well_separated_clusters: tuple[torch.Tensor, torch.Tensor],
        som_trained: SOM,
        device: str,
    ) -> None:
        """Test that the clustering metrics are calculated correctly."""
        data, labels = well_separated_clusters
        data = data.to(device)
        labels = labels.to(device)
        metrics_basic = calculate_clustering_metrics(data, labels)
        assert isinstance(metrics_basic, dict)
        assert "silhouette_score" in metrics_basic
        assert "davies_bouldin_score" in metrics_basic
        assert "calinski_harabasz_score" in metrics_basic
        assert "n_clusters" in metrics_basic
        assert "n_noise_points" in metrics_basic
        assert "noise_ratio" in metrics_basic
        _ = calculate_clustering_metrics(data, labels, som=som_trained)


class TestErrorCalculations:
    """Quantization and topographic error tests using SOM methods."""

    def test_quantization_error_single_sample(
        self,
        som_trained: SOM,
        single_sample: torch.Tensor,
    ) -> None:
        """Test quantization error calculation for single sample."""
        sample = single_sample.to(som_trained.device)
        qe = som_trained.quantization_error(sample)
        assert isinstance(qe, float)
        assert qe >= 0.0

    def test_quantization_error_batch(
        self,
        som_trained: SOM,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test quantization error calculation for batch."""
        data = small_random_data.to(som_trained.device)
        qe = som_trained.quantization_error(data)
        assert isinstance(qe, float)
        assert qe >= 0.0

    def test_topographic_error_calculation(
        self,
        som_trained: SOM,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test topographic error calculation."""
        data = small_random_data.to(som_trained.device)
        te = som_trained.topographic_error(data)
        assert isinstance(te, float)
        assert 0.0 <= te <= 1.0  # Topographic error is a ratio

    def test_error_consistency(
        self,
        som_trained: SOM,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test that error calculations are consistent across calls."""
        data = small_random_data.to(som_trained.device)
        qe1 = som_trained.quantization_error(data)
        qe2 = som_trained.quantization_error(data)
        te1 = som_trained.topographic_error(data)
        te2 = som_trained.topographic_error(data)
        assert qe1 == qe2
        assert te1 == te2
