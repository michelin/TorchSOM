"""Comprehensive unit tests for the SOM class."""

import warnings
from typing import Any

import pytest
import torch

from torchsom.core.som import SOM
from torchsom.utils.decay import DECAY_FUNCTIONS
from torchsom.utils.distances import DISTANCE_FUNCTIONS

pytestmark = [
    pytest.mark.unit,
]


class TestPrecomputations:
    def test_coordinate_distance_cache_shape(
        self,
        som_small: SOM,
    ) -> None:
        """Test that the coordinate distance cache has the correct shape."""
        # coord_distances_sq should exist and be [x*y, x*y]
        cache = som_small.coord_distances_sq
        n = som_small.x * som_small.y
        assert cache.shape == (n, n)
        assert torch.all(cache >= 0)

    def test_neighbor_offsets_exist(
        self,
        som_small: SOM,
    ) -> None:
        """Test that the neighbor offsets are precomputed."""
        # Offsets should be precomputed
        if som_small.topology == "hexagonal":
            assert hasattr(som_small, "_even_row_offsets")
            assert hasattr(som_small, "_odd_row_offsets")
            assert len(som_small._even_row_offsets) > 0
            assert len(som_small._odd_row_offsets) > 0
        else:
            assert hasattr(som_small, "_neighbor_offsets")
            assert len(som_small._neighbor_offsets) > 0

    def test_decay_schedules_length(
        self,
        som_small: SOM,
    ) -> None:
        """Test that the learning rate and sigma schedules have the correct length."""
        lr_schedule, sigma_schedule = som_small.lr_schedule, som_small.sigma_schedule
        assert lr_schedule.shape[0] == som_small.epochs
        assert sigma_schedule.shape[0] == som_small.epochs


class TestVectorizedNeighborhood:
    def test_vectorized_neighborhood_shapes(
        self,
        som_small: SOM,
    ) -> None:
        """Test that the vectorized neighborhood has the correct shape."""
        batch_size = 4
        rows = torch.randint(0, som_small.x, (batch_size,), device=som_small.device)
        cols = torch.randint(0, som_small.y, (batch_size,), device=som_small.device)
        bmu_indices_flat = rows * som_small.y + cols
        neighborhoods = som_small._vectorized_neighborhood(
            bmu_indices_flat=bmu_indices_flat, sigma=float(som_small.sigma)
        )
        assert neighborhoods.shape == (batch_size, som_small.x, som_small.y)
        assert torch.all(neighborhoods >= 0)

    def test_update_weights_runs(
        self,
        som_small: SOM,
    ) -> None:
        """Test that the weights are updated correctly."""
        original = som_small.weights.clone()
        data = torch.randn(6, som_small.num_features, device=som_small.device)
        bmus = som_small.identify_bmus(data)
        if bmus.dim() == 1:
            bmus = bmus.unsqueeze(0)
        lr = float(som_small.lr_schedule[0].item())
        sigma = float(som_small.sigma_schedule[0].item())
        som_small._update_weights(data, bmus, lr, sigma)
        assert not torch.allclose(original, som_small.weights)


class TestInputValidation:
    """Test input validation and edge cases."""

    def test_wrong_feature_dimension_raises_error(
        self,
        som_small: SOM,
    ) -> None:
        """Test that wrong feature dimensions are handled appropriately."""
        wrong_data = torch.randn(10, som_small.num_features + 1).to(som_small.device)
        with pytest.raises(RuntimeError):
            som_small.fit(wrong_data)

    @pytest.mark.xfail(
        strict=False,
        reason="SOM.fit does not currently support training with a single data point.",
    )
    def test_single_point_dataset(
        self,
        som_small: SOM,
    ) -> None:
        """Training with a single data point should complete and return errors.

        The current implementation supports batch size 1 during training.
        """
        single_point = torch.randn(1, som_small.num_features).to(som_small.device)
        q_errors, t_errors = som_small.fit(single_point)
        assert len(q_errors) == som_small.epochs
        assert len(t_errors) == som_small.epochs

    """
    NOTE: The current SOM implementation does not explicitly handle NaN or infinite values in input data.
    These tests are forward-looking: they will fail if SOM.fit does not raise an error
    For now, these tests are expected to fail (xfail).
    """

    # TODO: Implement explicit checks for NaN in SOM.fit and raise ValueError.
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


class TestSOMInitialization:
    """Test SOM constructor and parameter validation."""

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
        assert (
            som.neighborhood_fn_name
            == som_config_comprehensive["neighborhood_function"]
        )
        assert (
            som.distance_fn
            == DISTANCE_FUNCTIONS[som_config_comprehensive["distance_function"]]
        )
        assert (
            som.initialization_mode == som_config_comprehensive["initialization_mode"]
        )
        assert som.device == som_config_comprehensive["device"]
        assert som.random_seed == som_config_comprehensive["random_seed"]

    def test_invalid_topology_raises_error(
        self,
    ) -> None:
        """Test that invalid topology raises ValueError."""
        with pytest.raises(
            ValueError, match="Only hexagonal and rectangular topologies are supported"
        ):
            SOM(x=5, y=5, num_features=4, topology="invalid_topology")

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
        for batch_size in [8, len(data) + 16]:
            som = som_small
            som.batch_size = batch_size
            q_errors, t_errors = som.fit(data)
            assert len(q_errors) == som.epochs
            assert len(t_errors) == som.epochs

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

    def test_fit_empty_data_raises_error(
        self,
        som_small: SOM,
    ) -> None:
        """Test that fit method handles empty data appropriately."""
        pytest.xfail(
            reason="SOM.fit does not currently validate for empty datasets explicitly."
        )
        empty_data = torch.empty(0, som_small.num_features).to(som_small.device)
        som_small.fit(empty_data)


class TestBMUIdentification:
    """Test Best Matching Unit identification functionality."""

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


class TestCollectSamples:
    """Tests for SOM.collect_samples method."""

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
