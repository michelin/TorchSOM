"""Comprehensive unit tests for the SOM class.

Tests cover:
- SOM initialization and parameter validation
- Training functionality (fit method)
- BMU identification for single samples and batches
- Error calculations (quantization and topographic)
- Weight initialization methods
- Device compatibility (CPU/GPU)
- Input validation and edge cases such as NaN, Inf, empty data, only one point
- Map building functionality
"""

import warnings
from typing import Any

import pytest
import torch

from torchsom.core.som import SOM
from torchsom.utils.decay import DECAY_FUNCTIONS
from torchsom.utils.distances import DISTANCE_FUNCTIONS


class TestSOMInitialization:
    """Test SOM constructor and parameter validation."""

    # ! All parameters are fixtures parametrized in conftest.py, testing all combinations
    # ! ANother option would be to use @pytest.mark.parametrize
    @pytest.mark.unit
    def test_initialization_with_all_parameters(
        self,
        som_config_comprehensive: dict[str, Any],
    ) -> None:
        """Test SOM initialization with all parameters specified."""
        som = SOM(**som_config_comprehensive)

        assert som.x == som_config_comprehensive["x"]
        assert som.y == som_config_comprehensive["y"]
        assert som.num_features == som_config_comprehensive["num_features"]
        assert som.epochs == som_config_comprehensive["epochs"]
        assert som.batch_size == som_config_comprehensive["batch_size"]
        assert som.sigma == som_config_comprehensive["sigma"]
        assert som.learning_rate == som_config_comprehensive["learning_rate"]
        assert som.neighborhood_order == som_config_comprehensive["neighborhood_order"]
        assert som.topology == som_config_comprehensive["topology"]
        assert (
            som.lr_decay_fn
            == DECAY_FUNCTIONS[som_config_comprehensive["lr_decay_function"]]
        )
        assert (
            som.sigma_decay_fn
            == DECAY_FUNCTIONS[som_config_comprehensive["sigma_decay_function"]]
        )
        # assert som.neighborhood_fn == NEIGHBORHOOD_FUNCTIONS[neighborhood_function] # ! neighborhood_fn is a closure, so we can't compare it directly (wrapped in a lambda)
        # assert som.neighborhood_fn_name == neighborhood_function # ! Not implemented yet
        assert (
            som.distance_fn
            == DISTANCE_FUNCTIONS[som_config_comprehensive["distance_function"]]
        )
        assert (
            som.initialization_mode == som_config_comprehensive["initialization_mode"]
        )
        assert som.device == som_config_comprehensive["device"]
        assert som.random_seed == som_config_comprehensive["random_seed"]

    @pytest.mark.unit
    def test_invalid_topology_raises_error(
        self,
    ) -> None:
        """Test that invalid topology raises ValueError."""
        with pytest.raises(
            ValueError, match="Only hexagonal and rectangular topologies are supported"
        ):
            SOM(x=5, y=5, num_features=4, topology="invalid_topology")

    @pytest.mark.unit
    def test_invalid_distance_function_raises_error(
        self,
    ) -> None:
        """Test that invalid topology raises ValueError."""
        with pytest.raises(ValueError, match="Invalid distance function"):
            SOM(
                x=5,
                y=5,
                num_features=4,
                distance_function="invalid_distance_function",
            )

    @pytest.mark.unit
    def test_invalid_neighborhood_function_raises_error(
        self,
    ) -> None:
        """Test that invalid neighborhood function raises ValueError."""
        with pytest.raises(ValueError, match="Invalid neighborhood function"):
            SOM(
                x=5,
                y=5,
                num_features=4,
                neighborhood_function="invalid_neighborhood_function",
            )

    @pytest.mark.unit
    def test_invalid_lr_decay_function_raises_error(
        self,
    ) -> None:
        """Test that invalid learning rate decay function raises ValueError."""
        with pytest.raises(ValueError, match="Invalid learning rate decay function"):
            SOM(
                x=5,
                y=5,
                num_features=4,
                lr_decay_function="invalid_lr_decay_function",
            )

    @pytest.mark.unit
    def test_invalid_sigma_decay_function_raises_error(
        self,
    ) -> None:
        """Test that invalid sigma decay function raises ValueError."""
        with pytest.raises(ValueError, match="Invalid sigma decay function"):
            SOM(
                x=5,
                y=5,
                num_features=4,
                sigma_decay_function="invalid_sigma_decay_function",
            )

    @pytest.mark.unit
    def test_high_sigma_warning(
        self,
    ) -> None:
        """
        Test that initializing a SOM with a sigma value much larger than the map dimensions
        triggers a user warning.

        Rationale:
            - The SOM implementation emits a warning if the neighborhood width parameter (sigma)
              is set to a value that is likely too large for the given map size.
            - This test ensures that such a warning is actually raised, which helps users avoid
              unintentionally setting hyperparameters that could degrade SOM training quality.

        Test Steps:
            1. Use Python's warnings context manager to capture all warnings.
            2. Set the filter to always catch warnings, even if they have been triggered before.
            3. Instantiate a SOM with a very high sigma (10.0) on a small map (3x3).
            4. Assert that exactly one warning is raised.
            5. Assert that the warning message contains the expected substring indicating
               the sigma is too high.

        This test does not check for the type of warning (e.g., UserWarning), but focuses on
        the presence and content of the warning message.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SOM(x=3, y=3, num_features=2, sigma=10.0)
            assert len(w) == 1
            assert "sigma might be too high" in str(w[0].message)

    @pytest.mark.unit
    def test_weights_normalization(
        self,
        topology: str,
        device: str,
        initialization_mode: str,
        fixed_seed: int,
    ) -> None:
        """Test that initial weights are properly normalized (unit vectors)."""
        som = SOM(
            x=3,
            y=3,
            num_features=2,
            device=device,
            random_seed=fixed_seed,
            topology=topology,
            initialization_mode=initialization_mode,
        )
        weight_norms = torch.norm(som.weights, dim=-1)
        expected_norms = torch.ones_like(weight_norms)
        torch.testing.assert_close(weight_norms, expected_norms, atol=1e-5, rtol=1e-5)


class TestSOMTraining:
    """Test SOM training functionality."""

    @pytest.mark.unit
    def test_fit_returns_error_lists(
        self,
        som_small: SOM,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test that fit method returns quantization and topographic error lists."""
        data = small_random_data.to(som_small.device)
        q_errors, t_errors = som_small.fit(data)
        assert isinstance(q_errors, list)
        assert isinstance(t_errors, list)
        assert len(q_errors) == som_small.epochs
        assert len(t_errors) == som_small.epochs
        assert all(error >= 0 for error in q_errors)
        assert all(error >= 0 for error in t_errors)

    @pytest.mark.unit
    def test_fit_comprehensive(
        self,
        som_comprehensive: SOM,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test fit method with comprehensive configuration."""
        data = small_random_data.to(som_comprehensive.device)
        q_errors, t_errors = som_comprehensive.fit(data)

        assert isinstance(q_errors, list)
        assert isinstance(t_errors, list)
        assert len(q_errors) == som_comprehensive.epochs
        assert len(t_errors) == som_comprehensive.epochs
        assert all(error >= 0 for error in q_errors)
        assert all(error >= 0 for error in t_errors)

    @pytest.mark.unit
    def test_fit_with_different_batch_sizes(
        self,
        device: str,
        som_small: SOM,
        medium_random_data: torch.Tensor,
    ) -> None:
        """Test training with different batch sizes.

        Note:
            Batch size of 1 is excluded as it doesn't work for training.
            Batch size of len(data) + 16 is included to check behaviour when having a batch size larger than the data size.
        """
        data = medium_random_data.to(device)
        for batch_size in [8, 32, len(data), len(data) + 16]:
            som = som_small
            som.batch_size = batch_size
            q_errors, t_errors = som.fit(data)
            assert len(q_errors) == som.epochs
            assert len(t_errors) == som.epochs

    @pytest.mark.unit
    def test_training_convergence_direction(
        self,
        som_standard: SOM,
        medium_random_data: torch.Tensor,
    ) -> None:
        """Test that training generally improves (decreases) quantization error."""
        data = medium_random_data.to(som_standard.device)
        q_errors, _ = som_standard.fit(data)
        initial_avg = sum(q_errors[:5]) / 5
        final_avg = sum(q_errors[-5:]) / 5
        assert final_avg < initial_avg

    @pytest.mark.unit
    def test_fit_empty_data_raises_error(
        self,
        som_small: SOM,
    ) -> None:
        """Test that fit method handles empty data appropriately."""
        empty_data = torch.empty(0, som_small.num_features).to(som_small.device)
        with pytest.raises(ValueError):
            som_small.fit(empty_data)


class TestBMUIdentification:
    """Test Best Matching Unit identification functionality."""

    @pytest.mark.unit
    def test_identify_bmus_single_sample(
        self,
        som_trained: SOM,
        single_sample: torch.Tensor,
    ) -> None:
        """Test BMU identification for a single sample."""
        sample = single_sample.to(som_trained.device)
        bmu = som_trained.identify_bmus(sample)

        assert bmu.shape == (2,)  # Should return [row, col]
        assert 0 <= bmu[0] < som_trained.x
        assert 0 <= bmu[1] < som_trained.y
        assert bmu.dtype == torch.long

    @pytest.mark.unit
    def test_identify_bmus_batch(
        self,
        som_trained: SOM,
        medium_random_data: torch.Tensor,
    ) -> None:
        """Test BMU identification for a batch of samples."""
        data = medium_random_data.to(som_trained.device)
        bmus = som_trained.identify_bmus(data)

        assert bmus.shape == (data.shape[0], 2)  # Should return [batch_size, 2]
        assert torch.all(bmus[:, 0] >= 0) and torch.all(bmus[:, 0] < som_trained.x)
        assert torch.all(bmus[:, 1] >= 0) and torch.all(bmus[:, 1] < som_trained.y)
        assert bmus.dtype == torch.long

    @pytest.mark.unit
    def test_bmu_deterministic(
        self,
        som_small: SOM,
    ) -> None:
        """Test that BMU identification is deterministic for same input."""
        sample = torch.randn(som_small.num_features).to(som_small.device)
        bmu1 = som_small.identify_bmus(sample)
        bmu2 = som_small.identify_bmus(sample)

        # torch.testing.assert_close(bmu1, bmu2)
        assert torch.equal(bmu1, bmu2)

    @pytest.mark.unit
    def test_bmu_different_device_compatibility(
        self,
        som_small: SOM,
    ) -> None:
        """Test BMU identification when data is on different device than SOM."""
        # Create sample on CPU
        sample = torch.randn(som_small.num_features)

        # Should work regardless of device mismatch (method should handle it)
        bmu = som_small.identify_bmus(sample)
        assert bmu.shape == (2,)


class TestErrorCalculations:
    """Test quantization and topographic error calculations."""

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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


class TestWeightInitialization:
    """Test weight initialization methods."""

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_initialize_weights_invalid_mode(
        self,
        som_small: SOM,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test that invalid initialization mode raises error."""
        data = small_random_data.to(som_small.device)
        with pytest.raises(ValueError):
            som_small.initialize_weights(data, mode="invalid_mode")


class TestInputValidation:
    """Test input validation and edge cases."""

    @pytest.mark.unit
    def test_wrong_feature_dimension_raises_error(
        self,
        som_small: SOM,
    ) -> None:
        """Test that wrong feature dimensions are handled appropriately."""
        wrong_data = torch.randn(10, som_small.num_features + 1).to(som_small.device)
        with pytest.raises(RuntimeError):
            som_small.fit(wrong_data)

    # NOTE: The current SOM implementation does not explicitly handle NaN or infinite values in input data.
    # These tests are forward-looking: they will fail if SOM.fit does not raise an error
    # For now, these tests are expected to fail (xfail).

    # TODO: Implement explicit checks for NaN in SOM.fit and raise ValueError.
    @pytest.mark.unit
    @pytest.mark.xfail(
        strict=False,  # If the test fails, it won't cause a non-zero exit
        reason="SOM.fit does not currently validate for NaN values in input data.",
    )
    def test_nan_data_handling(
        self,
        som_small: SOM,
    ) -> None:
        """Test that SOM.fit raises an error when input data contains NaN values.

        This test is expected to fail until explicit NaN handling is implemented.
        """
        nan_data = torch.full((5, som_small.num_features), float("nan")).to(
            som_small.device
        )
        with pytest.raises((RuntimeError, ValueError)):
            som_small.fit(nan_data)

    # TODO: Implement explicit checks for Inf in SOM.fit and raise ValueError.
    @pytest.mark.unit
    @pytest.mark.xfail(
        strict=False,
        reason="SOM.fit does not currently validate for Inf values in input data.",
    )
    def test_inf_data_handling(
        self,
        som_small: SOM,
    ) -> None:
        """Test that SOM.fit raises an error when input data contains Inf values.

        This test is expected to fail until explicit Inf handling is implemented.
        """
        inf_data = torch.full((5, som_small.num_features), float("inf")).to(
            som_small.device
        )
        with pytest.raises((RuntimeError, ValueError)):
            som_small.fit(inf_data)

    @pytest.mark.unit
    def test_single_point_dataset(
        self,
        som_small: SOM,
    ) -> None:
        """Test that training with a single data point raises an error.

        SOM should not allow training on a dataset with only one sample,
        as this is insufficient for meaningful unsupervised learning.
        """
        single_point = torch.randn(1, som_small.num_features).to(som_small.device)
        with pytest.raises(TypeError):
            som_small.fit(single_point)


class TestDeviceCompatibility:
    """Test device compatibility and tensor operations."""

    @pytest.mark.unit
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "SOM weight initialization is not currently deterministic across consecutive instantiations, "
            "even with the same device and random seed. This is due to the random initialization logic in SOM."
        ),
    )
    def test_cpu_gpu_weight_consistency(
        self,
        fixed_seed: int,
    ) -> None:
        """Test that weights are consistent between CPU and GPU initialization.

        Note:
            This test is expected to fail until SOM weight initialization is made fully deterministic
            across devices and consecutive instantiations with the same random seed.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        som_cpu = SOM(x=5, y=5, num_features=4, device="cpu", random_seed=fixed_seed)
        som_gpu = SOM(x=5, y=5, num_features=4, device="cuda", random_seed=fixed_seed)

        # Weights should be equivalent (accounting for device)
        torch.testing.assert_close(som_cpu.weights, som_gpu.weights.cpu())

    @pytest.mark.gpu
    def test_gpu_memory_efficiency(
        self,
        som_config_standard: dict[str, Any],
    ) -> None:
        """Test GPU memory usage during training, it should be less than 25MB (not excessive)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        som = SOM(**som_config_standard)
        data = torch.randn(100, som.num_features, device=som.device)
        som.fit(data)
        final_memory = torch.cuda.memory_allocated()
        memory_used = final_memory - initial_memory

        assert memory_used < 25 * 1024 * 1024

    @pytest.mark.unit
    def test_device_transfer_in_methods(
        self,
        som_small: SOM,
        device: str,
    ) -> None:
        """Test that methods handle device transfer correctly."""
        data_cpu = torch.randn(10, som_small.num_features, device="cpu")
        bmus = som_small.identify_bmus(data_cpu)
        qe = som_small.quantization_error(data_cpu)
        te = som_small.topographic_error(data_cpu)

        assert bmus.device.type == device.split(":")[0]
        assert isinstance(qe, float)
        assert isinstance(te, float)


class TestMapBuilding:
    """Test various map building functionalities."""

    @pytest.mark.unit
    def test_build_hit_map(
        self,
        som_trained: SOM,
        small_random_data: torch.Tensor,
    ) -> None:
        """Test hit map generation."""
        data = small_random_data.to(som_trained.device)
        hit_map = som_trained.build_hit_map(data)

        assert hit_map.shape == (som_trained.x, som_trained.y)
        assert torch.all(hit_map >= 0)  # Hit counts should be non-negative
        assert hit_map.sum() == len(data)  # Total hits should equal number of samples

    @pytest.mark.unit
    def test_build_distance_map(
        self,
        som_trained: SOM,
    ) -> None:
        """Test distance map (U-matrix) generation."""
        distance_map = som_trained.build_distance_map()

        assert distance_map.shape == (som_trained.x, som_trained.y)
        assert torch.all(distance_map >= 0)  # Distances should be non-negative

    @pytest.mark.unit
    def test_build_bmus_data_map(
        self,
        som_trained: SOM,
        small_random_data: torch.Tensor,
        return_indices: bool,
    ) -> None:
        """
        Test mapping of winning neurons (BMUs) to their corresponding data points or indices.

        The mapping is created in batches for memory efficiency. The hit map is built on CPU,
        but BMU calculations are performed on GPU if available.

        Args:
            data (torch.Tensor): Input data tensor [num_samples, num_features] or [num_features].
            return_indices (bool, optional): If True, return indices instead of data points. Defaults to False.
            batch_size (int, optional): Size of batches to process. Defaults to 1024.

        Returns:
            Dict[Tuple[int, int], Any]: Dictionary mapping BMUs to data samples or indices.
        """
        data = small_random_data.to(som_trained.device)
        bmu_map = som_trained.build_bmus_data_map(data, return_indices=return_indices)

        # The returned object should always be a dict mapping (row, col) -> list
        assert isinstance(bmu_map, dict)
        assert all(isinstance(key, tuple) and len(key) == 2 for key in bmu_map.keys())

        if return_indices:
            # Each value should be a list of integer indices
            for list_of_indices in bmu_map.values():
                assert isinstance(list_of_indices, list)
                assert all(isinstance(idx, int) for idx in list_of_indices)
        else:
            # Each value should be a list of data points (torch.Tensor)
            for list_of_data_points in bmu_map.values():
                assert isinstance(list_of_data_points, torch.Tensor)
                # Each element should be a tensor of shape [num_features]
                for sample in list_of_data_points:
                    assert isinstance(sample, torch.Tensor)
                    assert sample.shape == (som_trained.num_features,)
                    # assert sample.device == som_trained.device

    @pytest.mark.unit
    def test_build_metric_map(
        self,
        som_trained: SOM,
        regression_data: tuple[torch.Tensor, torch.Tensor],
        reduction_parameter: str,
    ) -> None:
        """Test metric map generation."""
        data, target = regression_data
        data = data.to(som_trained.device)
        target = target.to(som_trained.device)
        metric_map = som_trained.build_metric_map(
            data, target=target, reduction_parameter=reduction_parameter
        )

        assert metric_map.shape == (som_trained.x, som_trained.y)

    @pytest.mark.unit
    def test_build_score_map(
        self,
        som_trained: SOM,
        regression_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test score map generation."""
        data, target = regression_data
        data = data.to(som_trained.device)
        target = target.to(som_trained.device)
        score_map = som_trained.build_score_map(data, target=target)

        non_nan_mask = ~torch.isnan(score_map)
        non_nan_score_map = score_map[non_nan_mask]

        assert score_map.shape == (som_trained.x, som_trained.y)
        assert torch.all(non_nan_score_map >= 0)  # Score values should be non-negative

    @pytest.mark.unit
    def test_build_rank_map(
        self,
        som_trained: SOM,
        regression_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test rank map generation."""
        data, target = regression_data
        data = data.to(som_trained.device)
        target = target.to(som_trained.device)
        rank_map = som_trained.build_rank_map(data, target=target)

        non_nan_mask = ~torch.isnan(rank_map)
        non_nan_rank_map = rank_map[non_nan_mask]

        assert rank_map.shape == (som_trained.x, som_trained.y)
        assert torch.all(non_nan_rank_map >= 0)  # Rank values should be non-negative

    @pytest.mark.unit
    def test_build_classification_map(
        self,
        som_trained: SOM,
        clustered_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test classification map generation."""
        data, target = clustered_data
        data = data.to(som_trained.device)
        target = target.to(som_trained.device)
        classification_map = som_trained.build_classification_map(data, target=target)

        non_nan_mask = ~torch.isnan(classification_map)
        non_nan_classification_map = classification_map[non_nan_mask]

        assert classification_map.shape == (som_trained.x, som_trained.y)
        assert torch.all(
            non_nan_classification_map >= 0
        )  # Classification values should be non-negative

    @pytest.mark.unit
    def test_map_building_consistency(
        self,
        som_trained: SOM,
        regression_data: tuple[torch.Tensor, torch.Tensor],
        clustered_data: tuple[torch.Tensor, torch.Tensor],
        return_indices: bool,
        reduction_parameter: str,
    ) -> None:
        """Test that map building is consistent across calls."""
        data, target = regression_data
        data = data.to(som_trained.device)
        target = target.to(som_trained.device)

        hit_map1 = som_trained.build_hit_map(data)
        hit_map2 = som_trained.build_hit_map(data)
        distance_map1 = som_trained.build_distance_map()
        distance_map2 = som_trained.build_distance_map()
        bmu_map1 = som_trained.build_bmus_data_map(data, return_indices=return_indices)
        bmu_map2 = som_trained.build_bmus_data_map(data, return_indices=return_indices)
        metric_map1 = som_trained.build_metric_map(
            data, target=target, reduction_parameter=reduction_parameter
        )
        metric_map2 = som_trained.build_metric_map(
            data, target=target, reduction_parameter=reduction_parameter
        )
        score_map1 = som_trained.build_score_map(data, target=target)
        score_map2 = som_trained.build_score_map(data, target=target)
        rank_map1 = som_trained.build_rank_map(data, target=target)
        rank_map2 = som_trained.build_rank_map(data, target=target)

        torch.testing.assert_close(hit_map1, hit_map2)
        torch.testing.assert_close(distance_map1, distance_map2)
        torch.testing.assert_close(bmu_map1, bmu_map2)
        torch.testing.assert_close(metric_map1, metric_map2, equal_nan=True)
        torch.testing.assert_close(score_map1, score_map2, equal_nan=True)
        torch.testing.assert_close(rank_map1, rank_map2, equal_nan=True)

        data, target = clustered_data
        data = data.to(som_trained.device)
        target = target.to(som_trained.device)

        classification_map1 = som_trained.build_classification_map(data, target=target)
        classification_map2 = som_trained.build_classification_map(data, target=target)

        torch.testing.assert_close(
            classification_map1, classification_map2, equal_nan=True
        )


class TestCollectSamples:
    """Tests for SOM.collect_samples method."""

    @pytest.mark.unit
    def test_collect_samples_basic_thresholding(
        self,
    ) -> None:
        """Collect samples from BMU bucket first, then nearest neighbor until threshold."""
        # Build a small deterministic SOM on CPU
        som = SOM(x=3, y=3, num_features=2, device="cpu", random_seed=0)

        # Manually set weights so that BMU at (1,1) is the query and (1,2) is the closest neighbor
        weights = torch.full((3, 3, 2), 10.0)
        weights[1, 1] = torch.tensor([0.0, 0.0])  # BMU
        weights[1, 2] = torch.tensor([0.1, 0.0])  # Closest neighbor
        som.weights.data = weights

        # Historical data: first column uniquely encodes the index
        historical_samples = torch.stack(
            [torch.tensor([float(i), 0.0]) for i in range(10)]
        )
        historical_outputs = torch.arange(10, dtype=torch.float32)

        # Map of BMUs to sample indices: BMU has [0], neighbor has [1,2]
        bmus_idx_map = {
            (1, 1): [0],
            (1, 2): [1, 2],
        }

        # Query equal to BMU weight guarantees BMU at (1,1)
        query = som.weights.data[1, 1].clone()

        data_buf, out_buf = som.collect_samples(
            query_sample=query,
            historical_samples=historical_samples,
            historical_outputs=historical_outputs,
            bmus_idx_map=bmus_idx_map,
            min_buffer_threshold=3,
        )

        # Expect exactly indices {0,1,2} gathered
        gathered_ids = set(data_buf[:, 0].tolist())
        assert gathered_ids == {0.0, 1.0, 2.0}
        assert data_buf.shape == (3, 2)
        assert out_buf.shape == (3, 1)

    @pytest.mark.unit
    def test_collect_samples_empty_bmu_uses_neighbors(
        self,
    ) -> None:
        """When BMU bucket is empty, it should pull from nearest neighbors via heap stage."""
        som = SOM(x=3, y=3, num_features=2, device="cpu", random_seed=0)

        # BMU weight at (1,1) and close neighbor at (1,2)
        weights = torch.full((3, 3, 2), 10.0)
        weights[1, 1] = torch.tensor([0.0, 0.0])
        weights[1, 2] = torch.tensor([0.1, 0.0])
        som.weights.data = weights

        historical_samples = torch.stack(
            [torch.tensor([float(i), 0.0]) for i in range(10)]
        )
        historical_outputs = torch.arange(10, dtype=torch.float32)

        # BMU bucket empty; neighbor has two indices
        bmus_idx_map = {
            (1, 1): [],
            (1, 2): [4, 5],
        }

        query = som.weights.data[1, 1].clone()
        data_buf, out_buf = som.collect_samples(
            query_sample=query,
            historical_samples=historical_samples,
            historical_outputs=historical_outputs,
            bmus_idx_map=bmus_idx_map,
            min_buffer_threshold=2,
        )

        gathered_ids = set(data_buf[:, 0].tolist())
        assert gathered_ids == {4.0, 5.0}
        assert out_buf.view(-1).tolist() == [4.0, 5.0]

    @pytest.mark.unit
    def test_collect_samples_no_available_samples_returns_empty(
        self,
        som_comprehensive: SOM,
    ) -> None:
        """Returns empty buffers when no BMU/neighbor neurons provide indices."""
        # Set BMU weight for determinism
        som_comprehensive.weights.data = torch.zeros_like(
            som_comprehensive.weights.data
        )
        som_comprehensive.weights.data[1, 1] = torch.zeros(
            som_comprehensive.num_features
        )

        historical_samples = torch.randn(5, som_comprehensive.num_features)
        historical_outputs = torch.randn(5)

        bmus_idx_map = {}  # No samples anywhere

        query = som_comprehensive.weights.data[1, 1].clone()
        data_buf, out_buf = som_comprehensive.collect_samples(
            query_sample=query,
            historical_samples=historical_samples,
            historical_outputs=historical_outputs,
            bmus_idx_map=bmus_idx_map,
            min_buffer_threshold=3,
        )

        assert data_buf.numel() == 0
        assert out_buf.numel() == 0
