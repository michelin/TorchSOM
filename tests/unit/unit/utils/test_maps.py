"""Unit tests for unified map APIs using SOM.build_map and SOM.build_multiple_maps."""

import pytest
import torch

from torchsom.core.som import SOM

pytestmark = [
    pytest.mark.unit,
]


class TestUnifiedMapAPIs:
    """Tests for map building via the unified interfaces."""

    def test_build_hit_map_unified(
        self,
        som_trained: SOM,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test that the hit map is built correctly."""
        data = small_random_data.to(som_trained.device)
        hit_map = som_trained.build_map("hit", data=data)

        assert hit_map.shape == (som_trained.x, som_trained.y)
        assert torch.all(hit_map >= 0)
        assert hit_map.sum() == len(data)

    def test_build_distance_map_options(
        self,
        som_trained: SOM,
    ) -> None:
        """Test that the distance map is built correctly."""
        distance_map_sum = som_trained.build_map(
            "distance", scaling="sum", neighborhood_order=1
        )
        distance_map_mean = som_trained.build_map(
            "distance", scaling="mean", neighborhood_order=2
        )

        for dm in [distance_map_sum, distance_map_mean]:
            assert dm.shape == (som_trained.x, som_trained.y)
            assert torch.all(dm >= 0)
            assert dm.max() <= 1.0

        with pytest.raises(ValueError, match="scaling should be either"):
            som_trained.build_map("distance", scaling="invalid")

    def test_build_bmus_data_map_unified(
        self,
        som_trained: SOM,
        small_random_data: torch.Tensor,
        return_indices: bool,
    ) -> None:
        """Test that the bmus data map is built correctly."""
        data = small_random_data.to(som_trained.device)
        bmus_map = som_trained.build_map(
            "bmus_data", data=data, return_indices=return_indices
        )

        assert isinstance(bmus_map, dict)
        assert all(isinstance(k, tuple) and len(k) == 2 for k in bmus_map.keys())
        if return_indices:
            for lst in bmus_map.values():
                assert isinstance(lst, list)
                assert all(isinstance(i, int) for i in lst)
        else:
            for tensor in bmus_map.values():
                assert isinstance(tensor, torch.Tensor)
                if tensor.numel() > 0:
                    assert tensor.shape[-1] == som_trained.num_features

    def test_build_metric_score_rank_maps_unified(
        self,
        som_trained: SOM,
        regression_data: tuple[torch.Tensor, torch.Tensor],
        reduction_parameter: str,
    ) -> None:
        """Test that the metric, score, and rank maps are built correctly."""
        data, target = regression_data
        data = data.to(som_trained.device)
        target = target.to(som_trained.device)

        metric_map = som_trained.build_map(
            "metric", data=data, target=target, reduction_parameter=reduction_parameter
        )
        score_map = som_trained.build_map("score", data=data, target=target)
        rank_map = som_trained.build_map("rank", data=data, target=target)

        for m in [metric_map, score_map, rank_map]:
            assert m.shape == (som_trained.x, som_trained.y)

        non_nan_mask = ~torch.isnan(score_map)
        if non_nan_mask.any():
            assert torch.all(score_map[non_nan_mask] >= 0)

        non_nan_mask = ~torch.isnan(rank_map)
        if non_nan_mask.any():
            assert torch.all(rank_map[non_nan_mask] >= 0)

    def test_build_classification_map_unified(
        self,
        som_trained: SOM,
        clustered_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that the classification map is built correctly."""
        data, labels = clustered_data
        data = data.to(som_trained.device)
        labels = labels.to(som_trained.device)

        classification_map = som_trained.build_map(
            "classification", data=data, target=labels, neighborhood_order=1
        )

        assert classification_map.shape == (som_trained.x, som_trained.y)
        non_nan_mask = ~torch.isnan(classification_map)
        if non_nan_mask.any():
            assert torch.all(classification_map[non_nan_mask] >= 0)

    def test_build_multiple_maps_unified(
        self,
        som_trained: SOM,
        regression_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that the multiple maps are built correctly."""
        data, target = regression_data
        data = data.to(som_trained.device)
        target = target.to(som_trained.device)

        map_configs = [
            {"type": "hit"},
            {"type": "metric", "kwargs": {"reduction_parameter": "std"}},
            {"type": "rank"},
            {"type": "classification", "kwargs": {"neighborhood_order": 2}},
        ]
        results = som_trained.build_multiple_maps(map_configs, data=data, target=target)

        assert isinstance(results, dict)
        assert len(results) == len(map_configs)
        for value in results.values():
            assert isinstance(value, torch.Tensor)
            assert value.shape == (som_trained.x, som_trained.y)


class TestMapInputValidation:
    """Focused tests for build_map input validation and errors."""

    def test_invalid_map_type_raises(self, som_trained: SOM) -> None:
        """Test that an invalid map type raises an error."""
        with pytest.raises(ValueError, match="Invalid map_type"):
            som_trained.build_map("unknown")

    def test_missing_target_raises(
        self, som_trained: SOM, small_random_data: torch.Tensor
    ) -> None:
        """Test that a missing target raises an error."""
        data = small_random_data.to(som_trained.device)
        with pytest.raises(ValueError, match="requires target"):
            som_trained.build_map("metric", data=data)

    def test_missing_data_raises(
        self, som_trained: SOM, regression_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that a missing data parameter raises an error."""
        _, target = regression_data
        target = target.to(som_trained.device)
        with pytest.raises(ValueError, match="requires data parameter"):
            som_trained.build_map("hit")

    def test_missing_bmus_source_raises(
        self, som_trained: SOM, regression_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that a missing bmus data map raises an error."""
        _, target = regression_data
        target = target.to(som_trained.device)
        with pytest.raises(ValueError, match="requires either data or bmus_data_map"):
            som_trained.build_map("metric", target=target)
