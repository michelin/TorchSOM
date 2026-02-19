"""Tests for BMU search strategies in torchsom.utils.search."""

from unittest.mock import patch

import pytest
import torch

from torchsom.utils.distances import DISTANCE_FUNCTIONS
from torchsom.utils.search import (
    FAISS_AVAILABLE,
    BMUSearchStrategy,
    FAISSSearch,
    TorchBruteForceSearch,
    create_search_strategy,
)

pytestmark = [
    pytest.mark.unit,
]


class TestTorchBruteForceSearch:
    """Tests for the default PyTorch brute-force search backend."""

    def test_search_returns_correct_shapes(self) -> None:
        """Search output tensors should have shape (batch, k)."""
        distance_fn = DISTANCE_FUNCTIONS["euclidean"]
        strategy = TorchBruteForceSearch(distance_fn)
        data = torch.randn(10, 4)
        weights = torch.randn(5, 5, 4)

        distances, indices = strategy.search(data, weights, k=1)
        assert distances.shape == (10, 1)
        assert indices.shape == (10, 1)

    def test_search_k_greater_than_one(self) -> None:
        """Search with k>1 should return k neighbours per sample."""
        distance_fn = DISTANCE_FUNCTIONS["euclidean"]
        strategy = TorchBruteForceSearch(distance_fn)
        data = torch.randn(8, 4)
        weights = torch.randn(5, 5, 4)

        distances, indices = strategy.search(data, weights, k=3)
        assert distances.shape == (8, 3)
        assert indices.shape == (8, 3)

    def test_search_finds_nearest_neuron(self) -> None:
        """Search should identify the geometrically closest neuron."""
        distance_fn = DISTANCE_FUNCTIONS["euclidean"]
        strategy = TorchBruteForceSearch(distance_fn)

        weights = torch.zeros(3, 3, 2)
        weights[1, 1] = torch.tensor([1.0, 0.0])
        weights[0, 0] = torch.tensor([10.0, 10.0])

        data = torch.tensor([[1.0, 0.0]])
        _, indices = strategy.search(data, weights, k=1)
        assert indices[0, 0].item() == 4  # flat index of (1,1) in 3x3

    def test_indices_are_valid_flat_indices(self) -> None:
        """Returned indices must lie within the flattened grid range."""
        distance_fn = DISTANCE_FUNCTIONS["euclidean"]
        strategy = TorchBruteForceSearch(distance_fn)
        data = torch.randn(20, 4)
        weights = torch.randn(5, 6, 4)

        _, indices = strategy.search(data, weights, k=1)
        assert (indices >= 0).all()
        assert (indices < 30).all()

    def test_rebuild_index_is_noop(self) -> None:
        """rebuild_index on brute-force strategy should be a no-op."""
        distance_fn = DISTANCE_FUNCTIONS["euclidean"]
        strategy = TorchBruteForceSearch(distance_fn)
        strategy.rebuild_index(torch.randn(5, 5, 4))

    def test_distances_are_non_negative(self) -> None:
        """All returned distances must be non-negative."""
        distance_fn = DISTANCE_FUNCTIONS["euclidean"]
        strategy = TorchBruteForceSearch(distance_fn)
        data = torch.randn(10, 4)
        weights = torch.randn(5, 5, 4)

        distances, _ = strategy.search(data, weights, k=1)
        assert (distances >= 0).all()

    @pytest.mark.parametrize(
        "metric", ["euclidean", "cosine", "manhattan", "chebyshev"]
    )
    def test_works_with_all_distance_functions(self, metric: str) -> None:
        """Search should succeed for all supported distance metrics."""
        distance_fn = DISTANCE_FUNCTIONS[metric]
        strategy = TorchBruteForceSearch(distance_fn)
        data = torch.randn(5, 4)
        weights = torch.randn(3, 3, 4)

        distances, indices = strategy.search(data, weights, k=1)
        assert distances.shape == (5, 1)
        assert indices.shape == (5, 1)


class TestFAISSSearchFallback:
    """Tests for FAISS search with unsupported metrics (falls back to torch)."""

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    def test_manhattan_falls_back_to_torch(self) -> None:
        """Manhattan metric should silently fall back to PyTorch brute-force."""
        distance_fn = DISTANCE_FUNCTIONS["manhattan"]
        strategy = FAISSSearch(distance_fn=distance_fn, distance_fn_name="manhattan")
        data = torch.randn(5, 4)
        weights = torch.randn(3, 3, 4)

        distances, indices = strategy.search(data, weights, k=1)
        assert distances.shape == (5, 1)
        assert indices.shape == (5, 1)

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    def test_chebyshev_falls_back_to_torch(self) -> None:
        """Chebyshev metric should silently fall back to PyTorch brute-force."""
        distance_fn = DISTANCE_FUNCTIONS["chebyshev"]
        strategy = FAISSSearch(distance_fn=distance_fn, distance_fn_name="chebyshev")
        data = torch.randn(5, 4)
        weights = torch.randn(3, 3, 4)

        distances, _indices = strategy.search(data, weights, k=1)
        assert distances.shape == (5, 1)


class TestFAISSSearchNative:
    """Tests for FAISS search with natively supported metrics."""

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    def test_euclidean_search_shapes(self) -> None:
        """FAISS euclidean search output should match (batch, k) shape."""
        distance_fn = DISTANCE_FUNCTIONS["euclidean"]
        strategy = FAISSSearch(distance_fn=distance_fn, distance_fn_name="euclidean")
        data = torch.randn(10, 4)
        weights = torch.randn(5, 5, 4)

        distances, indices = strategy.search(data, weights, k=1)
        assert distances.shape == (10, 1)
        assert indices.shape == (10, 1)

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    def test_cosine_search_shapes(self) -> None:
        """FAISS cosine search output should match (batch, k) shape."""
        distance_fn = DISTANCE_FUNCTIONS["cosine"]
        strategy = FAISSSearch(distance_fn=distance_fn, distance_fn_name="cosine")
        data = torch.randn(10, 4)
        weights = torch.randn(5, 5, 4)

        distances, indices = strategy.search(data, weights, k=1)
        assert distances.shape == (10, 1)
        assert indices.shape == (10, 1)

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    def test_euclidean_finds_correct_bmu(self) -> None:
        """FAISS should identify the geometrically closest neuron."""
        distance_fn = DISTANCE_FUNCTIONS["euclidean"]
        strategy = FAISSSearch(distance_fn=distance_fn, distance_fn_name="euclidean")

        weights = torch.zeros(3, 3, 2)
        weights[1, 1] = torch.tensor([1.0, 0.0])
        data = torch.tensor([[1.0, 0.0]])

        _, indices = strategy.search(data, weights, k=1)
        assert indices[0, 0].item() == 4

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    def test_rebuild_index_updates_search(self) -> None:
        """Rebuilding the index after weight change should reflect new weights."""
        distance_fn = DISTANCE_FUNCTIONS["euclidean"]
        strategy = FAISSSearch(distance_fn=distance_fn, distance_fn_name="euclidean")
        weights = torch.randn(3, 3, 4)
        strategy.rebuild_index(weights)
        assert strategy._index is not None

        new_weights = torch.randn(3, 3, 4)
        strategy.rebuild_index(new_weights)

        data = torch.randn(2, 4)
        distances, _indices = strategy.search(data, new_weights, k=1)
        assert distances.shape == (2, 1)

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    def test_k_greater_than_one(self) -> None:
        """FAISS search with k>1 should return k neighbours per sample."""
        distance_fn = DISTANCE_FUNCTIONS["euclidean"]
        strategy = FAISSSearch(distance_fn=distance_fn, distance_fn_name="euclidean")
        data = torch.randn(5, 4)
        weights = torch.randn(4, 4, 4)

        distances, indices = strategy.search(data, weights, k=3)
        assert distances.shape == (5, 3)
        assert indices.shape == (5, 3)


class TestFAISSImportError:
    """Tests for graceful handling when FAISS is not installed."""

    def test_import_error_when_faiss_unavailable(self) -> None:
        """FAISSSearch should raise ImportError when faiss is not installed."""
        with patch("torchsom.utils.search.FAISS_AVAILABLE", False):
            with patch("torchsom.utils.search.faiss", None):
                with pytest.raises(ImportError, match="faiss is required"):
                    FAISSSearch(
                        distance_fn=DISTANCE_FUNCTIONS["euclidean"],
                        distance_fn_name="euclidean",
                    )


class TestCreateSearchStrategy:
    """Tests for the strategy factory function."""

    def test_torch_backend_returns_brute_force(self) -> None:
        """backend='torch' should always return TorchBruteForceSearch."""
        strategy = create_search_strategy(
            backend="torch",
            distance_fn=DISTANCE_FUNCTIONS["euclidean"],
            distance_fn_name="euclidean",
            n_neurons=25,
        )
        assert isinstance(strategy, TorchBruteForceSearch)

    def test_auto_without_faiss_returns_brute_force(self) -> None:
        """auto backend falls back to brute-force when FAISS is unavailable."""
        with patch("torchsom.utils.search.FAISS_AVAILABLE", False):
            strategy = create_search_strategy(
                backend="auto",
                distance_fn=DISTANCE_FUNCTIONS["euclidean"],
                distance_fn_name="euclidean",
                n_neurons=1000,
            )
            assert isinstance(strategy, TorchBruteForceSearch)

    def test_auto_with_incompatible_metric_returns_brute_force(self) -> None:
        """auto backend falls back to brute-force for FAISS-incompatible metrics."""
        strategy = create_search_strategy(
            backend="auto",
            distance_fn=DISTANCE_FUNCTIONS["manhattan"],
            distance_fn_name="manhattan",
            n_neurons=1000,
        )
        assert isinstance(strategy, TorchBruteForceSearch)

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    def test_auto_with_faiss_and_euclidean_returns_faiss(self) -> None:
        """auto backend selects FAISS for euclidean metric on a large enough grid."""
        strategy = create_search_strategy(
            backend="auto",
            distance_fn=DISTANCE_FUNCTIONS["euclidean"],
            distance_fn_name="euclidean",
            n_neurons=256,
        )
        assert isinstance(strategy, FAISSSearch)

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    def test_faiss_backend_returns_faiss(self) -> None:
        """backend='faiss' should always return FAISSSearch."""
        strategy = create_search_strategy(
            backend="faiss",
            distance_fn=DISTANCE_FUNCTIONS["euclidean"],
            distance_fn_name="euclidean",
            n_neurons=25,
        )
        assert isinstance(strategy, FAISSSearch)

    def test_strategy_is_abstract(self) -> None:
        """BMUSearchStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BMUSearchStrategy()  # type: ignore[abstract]
