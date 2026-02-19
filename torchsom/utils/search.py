"""BMU search strategies with optional FAISS acceleration.

Provides a strategy abstraction for Best Matching Unit (BMU) search,
allowing transparent switching between PyTorch brute-force and FAISS-backed
nearest-neighbor search.

FAISS natively supports L2 (Euclidean) and inner-product metrics.
Cosine distance is handled by normalizing vectors before inner-product search.
Manhattan and Chebyshev distances fall back to PyTorch brute-force.
"""

from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional

import torch

try:
    import faiss  # type: ignore[import-untyped]

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False


_FAISS_COMPATIBLE_METRICS = {"euclidean", "cosine"}


class BMUSearchStrategy(ABC):
    """Abstract interface for BMU search backends."""

    @abstractmethod
    def search(
        self,
        data: torch.Tensor,
        weights: torch.Tensor,
        k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the k nearest neurons for each data sample.

        Args:
            data: Input tensor of shape ``[batch_size, num_features]``.
            weights: SOM weight tensor of shape ``[x, y, num_features]``.
            k: Number of nearest neighbors to return.

        Returns:
            A ``(distances, indices)`` tuple where *distances* has shape
            ``[batch_size, k]`` and *indices* has shape ``[batch_size, k]``
            (flat neuron indices into the ``x*y`` grid).
        """

    @abstractmethod
    def rebuild_index(self, weights: torch.Tensor) -> None:
        """Rebuild the internal index after weight updates.

        Args:
            weights: Updated SOM weight tensor of shape ``[x, y, num_features]``.
        """


class TorchBruteForceSearch(BMUSearchStrategy):
    """Brute-force BMU search using PyTorch distance functions."""

    def __init__(
        self,
        distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self._distance_fn = distance_fn

    def search(
        self,
        data: torch.Tensor,
        weights: torch.Tensor,
        k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distances = self._distance_fn(data, weights)
        batch_size = distances.shape[0]
        distances_flat = distances.view(batch_size, -1)
        if k == 1:
            indices = torch.argmin(distances_flat, dim=1, keepdim=True)
        else:
            _, indices = torch.topk(distances_flat, k, dim=1, largest=False)
        gathered = torch.gather(distances_flat, 1, indices)
        return gathered, indices

    def rebuild_index(self, weights: torch.Tensor) -> None:
        pass


class FAISSSearch(BMUSearchStrategy):
    """FAISS-backed BMU search for accelerated nearest-neighbor lookup.

    Falls back to :class:`TorchBruteForceSearch` when the chosen distance
    metric is not natively supported by FAISS (i.e. anything other than
    Euclidean or cosine).
    """

    def __init__(
        self,
        distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        distance_fn_name: str,
        device: str = "cpu",
        index_type: Literal["flat", "ivf"] = "flat",
        nprobe: int = 8,
    ) -> None:
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for the FAISS search backend. "
                "Install it with: pip install faiss-cpu  (or faiss-gpu)"
            )

        self._distance_fn_name = distance_fn_name
        self._device = device
        self._index_type = index_type
        self._nprobe = nprobe
        self._index: Optional["faiss.Index"] = None
        self._use_cosine = distance_fn_name == "cosine"

        if distance_fn_name not in _FAISS_COMPATIBLE_METRICS:
            self._fallback = TorchBruteForceSearch(distance_fn)
        else:
            self._fallback = None

    def _build_index(self, vectors: torch.Tensor) -> "faiss.Index":
        """Build a FAISS index from a flat ``[n, d]`` float32 tensor."""
        n, d = vectors.shape
        vectors_np = vectors.detach().cpu().float().numpy()

        if self._use_cosine:
            faiss.normalize_L2(vectors_np)

        if self._index_type == "ivf" and n >= 256:
            nlist = min(int(n**0.5), n // 4)
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(vectors_np)
            index.nprobe = self._nprobe
        else:
            index = faiss.IndexFlatL2(d)

        index.add(vectors_np)

        if self._device.startswith("cuda") and hasattr(faiss, "index_cpu_to_gpu"):
            gpu_id = 0
            if ":" in self._device:
                gpu_id = int(self._device.split(":")[1])
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)

        return index

    def rebuild_index(self, weights: torch.Tensor) -> None:
        x, y, d = weights.shape
        flat_weights = weights.view(-1, d)
        if self._fallback is not None:
            return
        self._index = self._build_index(flat_weights)

    def search(
        self,
        data: torch.Tensor,
        weights: torch.Tensor,
        k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fallback is not None:
            return self._fallback.search(data, weights, k)

        if self._index is None:
            self.rebuild_index(weights)

        query = data.detach().cpu().float().numpy()
        if self._use_cosine:
            faiss.normalize_L2(query)

        distances_np, indices_np = self._index.search(query, k)

        distances = torch.from_numpy(distances_np).to(data.device)
        indices = torch.from_numpy(indices_np).long().to(data.device)

        if self._use_cosine:
            distances = torch.clamp(distances, min=0.0)

        return distances, indices


def create_search_strategy(
    backend: Literal["auto", "torch", "faiss"],
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    distance_fn_name: str,
    device: str = "cpu",
    faiss_index_type: Literal["flat", "ivf"] = "flat",
    faiss_nprobe: int = 8,
) -> BMUSearchStrategy:
    """Factory that instantiates the appropriate search strategy.

    Args:
        backend: ``"auto"`` selects FAISS when available and the metric is
            compatible, otherwise falls back to PyTorch.  ``"torch"`` and
            ``"faiss"`` force a specific backend.
        distance_fn: The PyTorch distance callable (used by the torch backend
            and as a FAISS fallback for unsupported metrics).
        distance_fn_name: Name of the distance function (e.g. ``"euclidean"``).
        device: Compute device (``"cpu"`` or ``"cuda"``).
        faiss_index_type: FAISS index structure, ``"flat"`` for exact search
            or ``"ivf"`` for approximate.
        faiss_nprobe: Number of cells to probe when using IVF indices.

    Returns:
        A concrete :class:`BMUSearchStrategy` instance.
    """
    if backend == "faiss" or (
        backend == "auto"
        and FAISS_AVAILABLE
        and distance_fn_name in _FAISS_COMPATIBLE_METRICS
    ):
        return FAISSSearch(
            distance_fn=distance_fn,
            distance_fn_name=distance_fn_name,
            device=device,
            index_type=faiss_index_type,
            nprobe=faiss_nprobe,
        )
    return TorchBruteForceSearch(distance_fn)
