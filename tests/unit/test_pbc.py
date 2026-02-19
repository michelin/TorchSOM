"""Tests for Periodic Boundary Conditions (PBC) in the SOM."""

import math

import pytest
import torch

from torchsom.core.som import SOM
from torchsom.utils.metrics import calculate_topographic_error

pytestmark = [
    pytest.mark.unit,
]


class TestPBCCoordinateDistances:
    """Verify that PBC wraps coordinate distances correctly."""

    def test_corner_neurons_are_close_with_pbc(self) -> None:
        """Opposite corners should be close on a toroidal grid."""
        som = SOM(x=10, y=10, num_features=4, pbc=True, device="cpu")
        idx_top_left = 0
        idx_bottom_right = 10 * 10 - 1

        dist_sq = som.coord_distances_sq[idx_top_left, idx_bottom_right].item()
        diag = math.sqrt(1.0**2 + 1.0**2)
        assert math.sqrt(dist_sq) <= diag + 0.5

    def test_edge_neurons_wrap_horizontally(self) -> None:
        """Left-edge and right-edge neurons on the same row should be neighbours."""
        som = SOM(x=6, y=6, num_features=4, pbc=True, device="cpu")
        left = 0 * 6 + 0
        right = 0 * 6 + 5

        dist_sq = som.coord_distances_sq[left, right].item()
        assert math.sqrt(dist_sq) <= 1.5

    def test_edge_neurons_wrap_vertically(self) -> None:
        """Top-row and bottom-row neurons in the same column should be neighbours."""
        som = SOM(x=6, y=6, num_features=4, pbc=True, device="cpu")
        top = 0 * 6 + 0
        bottom = 5 * 6 + 0

        dist_sq = som.coord_distances_sq[top, bottom].item()
        assert math.sqrt(dist_sq) <= 1.5

    def test_pbc_disabled_edges_are_far(self) -> None:
        """Without PBC, opposite edges should have large distance."""
        som = SOM(x=10, y=10, num_features=4, pbc=False, device="cpu")
        idx_0 = 0 * 10 + 0
        idx_far = 9 * 10 + 9

        dist_sq_no_pbc = som.coord_distances_sq[idx_0, idx_far].item()
        assert math.sqrt(dist_sq_no_pbc) > 10.0

    def test_pbc_symmetric_distances(self) -> None:
        """PBC distance matrix should remain symmetric."""
        som = SOM(x=8, y=8, num_features=4, pbc=True, device="cpu")
        diff = som.coord_distances_sq - som.coord_distances_sq.T
        assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-6)

    def test_self_distance_is_zero(self) -> None:
        """Distance from any neuron to itself should be zero."""
        som = SOM(x=6, y=6, num_features=4, pbc=True, device="cpu")
        diag = torch.diag(som.coord_distances_sq)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-7)


class TestPBCHexagonal:
    """PBC tests specific to hexagonal topology."""

    def test_hexagonal_pbc_corner_wrap(self) -> None:
        som = SOM(
            x=8, y=8, num_features=4, topology="hexagonal", pbc=True, device="cpu"
        )
        top_left = 0
        bottom_right = 8 * 8 - 1
        dist_sq = som.coord_distances_sq[top_left, bottom_right].item()
        max_non_pbc = math.sqrt(8.0**2 + (8.0 * math.sqrt(3) / 2) ** 2)
        assert math.sqrt(dist_sq) < max_non_pbc

    def test_hexagonal_pbc_symmetric(self) -> None:
        som = SOM(
            x=6, y=6, num_features=4, topology="hexagonal", pbc=True, device="cpu"
        )
        diff = som.coord_distances_sq - som.coord_distances_sq.T
        assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-6)


class TestPBCNeighborhood:
    """Verify that PBC neighborhood influence wraps correctly."""

    def test_corner_neuron_has_full_neighborhood(self) -> None:
        """With PBC a corner neuron should have the same neighbourhood
        influence sum as a centre neuron."""
        som = SOM(
            x=10, y=10, num_features=4, pbc=True,
            neighborhood_function="gaussian", sigma=2.0, device="cpu",
        )
        corner_idx = torch.tensor([0])
        center_idx = torch.tensor([5 * 10 + 5])

        corner_nb = som._vectorized_neighborhood(corner_idx, sigma=2.0)
        center_nb = som._vectorized_neighborhood(center_idx, sigma=2.0)

        assert torch.allclose(
            corner_nb.sum(), center_nb.sum(), atol=0.3
        )

    def test_without_pbc_corner_has_less_influence(self) -> None:
        """Without PBC, corner neighbourhood sums should be lower than centre."""
        som = SOM(
            x=10, y=10, num_features=4, pbc=False,
            neighborhood_function="gaussian", sigma=2.0, device="cpu",
        )
        corner_idx = torch.tensor([0])
        center_idx = torch.tensor([5 * 10 + 5])

        corner_nb = som._vectorized_neighborhood(corner_idx, sigma=2.0)
        center_nb = som._vectorized_neighborhood(center_idx, sigma=2.0)

        assert corner_nb.sum() < center_nb.sum()


class TestPBCTraining:
    """Verify that PBC-enabled SOMs train successfully."""

    def test_fit_runs_without_error(self) -> None:
        torch.manual_seed(42)
        data = torch.randn(100, 4)
        data = (data - data.mean(0)) / data.std(0)

        som = SOM(
            x=5, y=5, num_features=4, epochs=3, batch_size=16,
            pbc=True, device="cpu", random_seed=42,
        )
        q_errors, t_errors = som.fit(data, verbose=False)
        assert len(q_errors) == 3
        assert len(t_errors) == 3
        assert all(isinstance(e, float) for e in q_errors)

    def test_pbc_quantization_error_is_finite(self) -> None:
        torch.manual_seed(42)
        data = torch.randn(80, 4)
        data = (data - data.mean(0)) / data.std(0)

        som = SOM(
            x=5, y=5, num_features=4, epochs=3, batch_size=16,
            pbc=True, device="cpu", random_seed=42,
        )
        som.fit(data, verbose=False)
        qe = som.quantization_error(data)
        assert math.isfinite(qe)

    def test_pbc_hexagonal_training(self) -> None:
        torch.manual_seed(42)
        data = torch.randn(80, 4)
        data = (data - data.mean(0)) / data.std(0)

        som = SOM(
            x=5, y=5, num_features=4, epochs=3, batch_size=16,
            topology="hexagonal", pbc=True, device="cpu", random_seed=42,
        )
        q_errors, _ = som.fit(data, verbose=False)
        assert len(q_errors) == 3


class TestPBCTopographicError:
    """Verify topographic error handles PBC wrapping."""

    def test_topographic_error_with_pbc(self) -> None:
        torch.manual_seed(42)
        data = torch.randn(50, 4)
        weights = torch.randn(5, 5, 4)

        from torchsom.utils.distances import DISTANCE_FUNCTIONS

        te = calculate_topographic_error(
            data, weights, DISTANCE_FUNCTIONS["euclidean"],
            topology="rectangular", pbc=True,
        )
        assert 0.0 <= te <= 1.0

    def test_topographic_error_pbc_vs_no_pbc(self) -> None:
        """PBC should generally produce equal or lower topographic error
        because adjacency wraps around edges."""
        torch.manual_seed(42)
        data = torch.randn(100, 4)
        weights = torch.randn(5, 5, 4)

        from torchsom.utils.distances import DISTANCE_FUNCTIONS

        te_no_pbc = calculate_topographic_error(
            data, weights, DISTANCE_FUNCTIONS["euclidean"],
            topology="rectangular", pbc=False,
        )
        te_pbc = calculate_topographic_error(
            data, weights, DISTANCE_FUNCTIONS["euclidean"],
            topology="rectangular", pbc=True,
        )
        assert te_pbc <= te_no_pbc


class TestPBCCollectSamples:
    """Verify collect_samples wraps neighbours with PBC."""

    def test_collect_samples_wraps_with_pbc(self) -> None:
        torch.manual_seed(42)
        data = torch.randn(100, 4)
        data = (data - data.mean(0)) / data.std(0)

        som = SOM(
            x=5, y=5, num_features=4, epochs=2, batch_size=16,
            pbc=True, device="cpu", random_seed=42,
        )
        som.fit(data, verbose=False)

        bmus_map = som.build_map("bmus_data", data=data, return_indices=True)
        query = data[0]
        buf_data, buf_out = som.collect_samples(
            query_sample=query,
            historical_samples=data,
            historical_outputs=torch.randn(100),
            bmus_idx_map=bmus_map,
            min_buffer_threshold=5,
        )
        assert buf_data.shape[0] > 0
        assert buf_out.shape[0] == buf_data.shape[0]
