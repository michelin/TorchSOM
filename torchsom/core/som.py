"""PyTorch implementation of classic Self Organizing Maps using batch learning."""

import math
import warnings
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torchsom.core.base_som import BaseSOM
from torchsom.utils.clustering import cluster_data, extract_clustering_features
from torchsom.utils.decay import DECAY_FUNCTIONS
from torchsom.utils.distances import DISTANCE_FUNCTIONS
from torchsom.utils.grid import adjust_meshgrid_topology, create_mesh_grid
from torchsom.utils.initialization import initialize_weights
from torchsom.utils.maps import MAP_FUNCTIONS
from torchsom.utils.metrics import (
    calculate_clustering_metrics,
    calculate_quantization_error,
    calculate_topographic_error,
)
from torchsom.utils.neighborhood import NEIGHBORHOOD_FUNCTIONS
from torchsom.utils.search import create_search_strategy
from torchsom.utils.topology import get_all_neighbors_up_to_order


class SOM(BaseSOM):
    """PyTorch implementation of Self Organizing Maps using batch learning.

    Args:
        BaseSOM: Abstract base class for SOM variants
    """

    def __init__(
        self,
        x: int,
        y: int,
        num_features: int,
        epochs: int = 10,
        batch_size: int = 5,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        neighborhood_order: int = 1,
        topology: str = "rectangular",
        lr_decay_function: str = "asymptotic_decay",
        sigma_decay_function: str = "asymptotic_decay",
        neighborhood_function: str = "gaussian",
        distance_function: str = "euclidean",
        initialization_mode: str = "random",
        pbc: bool = False,
        search_backend: str = "auto",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        random_seed: int = 42,
    ):
        """Initialize the SOM.

        Args:
            x (int): Number of rows
            y (int): Number of cols
            num_features (int): Number of input features
            epochs (int, optional): Number of epochs to train. Defaults to 10.
            batch_size (int, optional): Number of samples to be considered at each epoch (training). Defaults to 5.
            sigma (float, optional): Width of the neighborhood, so standard deviation. It controls the spread of the update influence. Defaults to 1.0.
            learning_rate (float, optional): Strength of the weights updates. Defaults to 0.5.
            neighborhood_order (int, optional): Number of neighbors to consider for the distance calculation. Defaults to 1.
            topology (str, optional): Grid configuration. Defaults to "rectangular".
            lr_decay_function (str, optional): Function to adjust (decrease) the learning rate at each epoch (training). Defaults to "asymptotic_decay".
            sigma_decay_function (str, optional): Function to adjust (decrease) the sigma at each epoch (training). Defaults to "asymptotic_decay".
            neighborhood_function (str, optional): Function to update the weights at each epoch (training). Defaults to "gaussian".
            distance_function (str, optional): Function to compute the distance between grid weights and input data. Defaults to "euclidean".
            initialization_mode (str, optional): Method to initialize SOM weights. Defaults to "random".
            pbc (bool, optional): Enable periodic boundary conditions (toroidal topology). Defaults to False.
            search_backend (str, optional): BMU search backend. ``"auto"`` uses FAISS when available, ``"torch"`` forces PyTorch brute-force, ``"faiss"`` forces FAISS. Defaults to "auto".
            device (str, optional): Allocate tensors on CPU or GPU. Defaults to "cuda" if available, else "cpu".
            random_seed (int, optional): Ensure reproducibility. Defaults to 42.

        Raises:
            ValueError: Ensure valid topology
        """
        super().__init__()
        if sigma > torch.sqrt(torch.tensor(float(x * x + y * y))):
            warnings.warn(
                "Warning: sigma might be too high for the dimension of the map.",
                stacklevel=2,
            )
        if topology not in ["hexagonal", "rectangular"]:
            raise ValueError("Only hexagonal and rectangular topologies are supported")
        if lr_decay_function not in DECAY_FUNCTIONS:
            raise ValueError("Invalid learning rate decay function")
        if sigma_decay_function not in DECAY_FUNCTIONS:
            raise ValueError("Invalid sigma decay function")
        if distance_function not in DISTANCE_FUNCTIONS:
            raise ValueError("Invalid distance function")
        if neighborhood_function not in NEIGHBORHOOD_FUNCTIONS:
            raise ValueError("Invalid neighborhood function")
        if search_backend not in ("auto", "torch", "faiss"):
            raise ValueError("search_backend must be 'auto', 'torch', or 'faiss'")

        self.x = x
        self.y = y
        self.num_features = num_features
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.topology = topology
        self.pbc = pbc
        self.random_seed = random_seed
        self.neighborhood_order = neighborhood_order
        self.distance_fn_name = distance_function
        self.neighborhood_fn_name = neighborhood_function
        self.initialization_mode = initialization_mode
        self.distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            DISTANCE_FUNCTIONS[distance_function]
        )
        self.lr_decay_fn = DECAY_FUNCTIONS[lr_decay_function]
        self.sigma_decay_fn = DECAY_FUNCTIONS[sigma_decay_function]
        self._search_strategy = create_search_strategy(
            backend=search_backend,
            distance_fn=self.distance_fn,
            distance_fn_name=distance_function,
            device=device,
        )

        x_meshgrid, y_meshgrid = create_mesh_grid(x, y, device)
        self.xx, self.yy = adjust_meshgrid_topology(x_meshgrid, y_meshgrid, topology)

        torch.manual_seed(random_seed)

        weights = 2 * torch.randn(x, y, num_features, device=device) - 1
        normalized_weights = weights / torch.norm(weights, dim=-1, keepdim=True)
        self.weights = nn.Parameter(normalized_weights, requires_grad=False)

        self._precompute_coordinate_distances()
        self._precompute_neighbor_offsets()
        self.lr_schedule, self.sigma_schedule = self._precompute_decay_schedules(
            epochs=self.epochs
        )

    def _precompute_coordinate_distances(self) -> None:
        """Pre-compute coordinate distance matrices for all neuron pairs, used during neighborhood calculations.

        When ``pbc=True``, applies the minimum-image convention so that
        coordinate distances wrap around the grid boundaries (toroidal topology).
        """
        coords = torch.stack([self.xx.flatten(), self.yy.flatten()], dim=1)
        coord_diff = coords.unsqueeze(1) - coords.unsqueeze(0)

        if self.pbc:
            grid_size_x = float(self.x)
            grid_size_y = float(self.y)
            if self.topology == "hexagonal":
                grid_size_y *= math.sqrt(3) / 2
            grid_size = torch.tensor(
                [grid_size_x, grid_size_y], device=coords.device
            )
            coord_diff = coord_diff - grid_size * torch.round(
                coord_diff / grid_size
            )

        self.coord_distances_sq = torch.sum(coord_diff**2, dim=2)

    def _precompute_neighbor_offsets(self) -> None:
        """Precompute neighbor offsets for the SOM's topology and neighborhood order."""
        self._neighbor_offsets = get_all_neighbors_up_to_order(
            topology=self.topology,
            max_order=self.neighborhood_order,
        )
        if self.topology == "hexagonal":
            self._even_row_offsets = self._neighbor_offsets["even"]
            self._odd_row_offsets = self._neighbor_offsets["odd"]

    def _precompute_decay_schedules(
        self,
        epochs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute decay schedules for all epochs at once.

        Args:
            epochs (int): Number of epochs

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Learning rate and sigma schedules
        """
        epoch_indices = torch.arange(epochs, dtype=torch.float32)
        lr_schedule = torch.tensor(
            [
                self.lr_decay_fn(self.learning_rate, t=epoch, max_iter=epochs)
                for epoch in epoch_indices
            ]
        )
        sigma_schedule = torch.tensor(
            [
                self.sigma_decay_fn(self.sigma, t=epoch, max_iter=epochs)
                for epoch in epoch_indices
            ]
        )
        return lr_schedule, sigma_schedule

    def _vectorized_neighborhood(
        self,
        bmu_indices_flat: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Compute neighborhood weights using vectorized operations.

        Args:
            bmu_indices_flat (torch.Tensor): Flattened BMU indices [batch_size]
            sigma (float): Current sigma value

        Returns:
            torch.Tensor: Neighborhood weights [batch_size, x, y]
        """
        return NEIGHBORHOOD_FUNCTIONS[self.neighborhood_fn_name](
            self.coord_distances_sq, bmu_indices_flat, sigma, self.x, self.y
        )

    def _update_weights(
        self,
        data: torch.Tensor,  # [batch, features]
        bmus: torch.Tensor,  # [batch, 2]
        learning_rate: float,
        sigma: float,
    ) -> None:
        """Update weights using vectorized neighborhood calculations.

        Args:
            data (torch.Tensor): Input tensor of shape [batch_size, features]
            bmus (torch.Tensor): BMU coordinates as tensor [batch_size, 2]
            learning_rate (float): Current learning rate
            sigma (float): Current sigma value
        """
        batch_size = data.shape[0]
        # Convert BMU coordinates to flat indices for efficient lookup: [batch]
        bmu_indices_flat = bmus[:, 0] * self.y + bmus[:, 1]
        # Compute neighborhood weights efficiently using vectorized functions: [batch, x, y]
        neighborhoods = self._vectorized_neighborhood(bmu_indices_flat, sigma)
        # Compute neighborhood sum: [x, y]
        neighborhood_sum = neighborhoods.sum(dim=0)
        # Compute weighted data: [x, y, features]
        weighted_data = torch.einsum("bxy,bf->xyf", neighborhoods, data)
        # Compute updates: [x, y, features]
        updates = (weighted_data - neighborhood_sum.unsqueeze(-1) * self.weights) * (
            learning_rate / batch_size
        )
        self.weights.data += updates
        self._search_strategy.rebuild_index(self.weights)

    def _calculate_distances_to_neurons(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate distances between input data and all neurons.

        Args:
            data (torch.Tensor): Input tensor of shape [num_features] if single or [batch_size, num_features] if batch

        Returns:
            torch.Tensor: Distances tensor of shape [x, y] or [batch_size, x, y]
        """
        data = data.to(self.device)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        # Compute distances between data and all neurons: [batch_size, x, y]
        distances = self.distance_fn(data, self.weights)
        # Handle single sample case: [x, y]
        if distances.shape[0] == 1 and data.shape[0] == 1:
            distances = distances.squeeze(0)
        return distances

    def identify_bmus(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Find BMUs for input data.

        Uses the configured search strategy (PyTorch brute-force or FAISS).

        Args:
            data (torch.Tensor): Input tensor of shape [batch_size, features] or [features]

        Returns:
            torch.Tensor: BMU coordinates as tensor [batch_size, 2] or [2]
        """
        data = data.to(self.device)
        single = data.dim() == 1
        if single:
            data = data.unsqueeze(0)

        _, indices = self._search_strategy.search(data, self.weights, k=1)
        flat_indices = indices.squeeze(1)

        rows = torch.div(flat_indices, self.y, rounding_mode="floor")
        cols = flat_indices % self.y
        result = torch.stack([rows, cols], dim=1)

        if single:
            return result.squeeze(0)
        return result

    def quantization_error(
        self,
        data: torch.Tensor,
    ) -> float:
        """Calculate quantization error.

        Args:
            data (torch.Tensor): input data tensor [batch_size, num_features] or [num_features]

        Returns:
            float: Average quantization error value
        """
        data = data.to(self.device)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        return calculate_quantization_error(data, self.weights, self.distance_fn)

    def topographic_error(
        self,
        data: torch.Tensor,
    ) -> float:
        """Calculate topographic error with batch support.

        Args:
            data (torch.Tensor): input data tensor [batch_size, num_features] or [num_features]

        Returns:
            float: Topographic error ratio
        """
        data = data.to(self.device)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        return calculate_topographic_error(
            data, self.weights, self.distance_fn, self.topology, pbc=self.pbc
        )

    def initialize_weights(
        self,
        data: torch.Tensor,
        mode: Optional[str] = None,
    ) -> None:
        """Data should be normalized before initialization.

        Initialize weights using:

            1. Random samples from input data.
            2. PCA components to make the training process converge faster.

        Args:
            data (torch.Tensor): input data tensor [batch_size, num_features]
            mode (str, optional): selection of the method to init the weights. Defaults to None.

        Raises:
            ValueError: Ensure neurons' weights and input data have the same number of features
            RuntimeError: If random initialization takes too long
            ValueError: Requires at least 2 features for PCA
            ValueError: Requires more than one sample to perform PCA
            ValueError: Ensure an appropriate method for initialization
        """
        data = data.to(self.device)
        if data.shape[1] != self.num_features:
            raise ValueError(
                f"Input data dimension ({data.shape[1]}) and weights dimension ({self.num_features}) don't match"
            )
        if mode is None:
            mode = self.initialization_mode
        new_weights = initialize_weights(
            weights=self.weights.data,
            data=data,
            mode=mode,
            topology=self.topology,
            device=self.device,
        )
        self.weights.data = new_weights

    def fit(
        self,
        data: torch.Tensor,
        verbose: bool = True,
    ) -> tuple[list[float], list[float]]:
        """Train the SOM using batches and track errors.

        Args:
            data (torch.Tensor): input data tensor [batch_size, num_features]
            verbose (bool, optional): Whether to print progress. Defaults to True.

        Returns:
            Tuple[List[float], List[float]]: Quantization and topographic errors [epoch]
        """
        dataset = TensorDataset(data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
        )

        epoch_q_errors = []
        epoch_t_errors = []
        for epoch in tqdm(
            range(self.epochs),
            desc="Training SOM",
            unit="epoch",
            disable=not verbose,
        ):
            # Update learning parameters thanks to decay functions (schedulers)
            lr = self.lr_schedule[epoch]
            sigma = self.sigma_schedule[epoch]
            batch_q_errors = []
            batch_t_errors = []
            for batch in dataloader:
                batch_data = batch[0].to(self.device)
                # Rerieve the BMUs [batch_size, 2]
                bmus = self.identify_bmus(batch_data)
                # Update the weights
                self._update_weights(batch_data, bmus, lr, sigma)
                # Calculate batch errors
                batch_q_errors.append(self.quantization_error(batch_data))
                batch_t_errors.append(self.topographic_error(batch_data))
            # Calculate average epoch errors
            epoch_q_errors.append(torch.tensor(batch_q_errors).mean().item())
            epoch_t_errors.append(100 * torch.tensor(batch_t_errors).mean().item())

        return epoch_q_errors, epoch_t_errors

    def collect_samples(
        self,
        query_sample: torch.Tensor,
        historical_samples: torch.Tensor,
        historical_outputs: torch.Tensor,
        bmus_idx_map: dict[tuple[int, int], list[int]],
        min_buffer_threshold: int = 50,
        return_indices: bool = False,
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Collect historical samples similar to the query sample using SOM projection.

        Args:
            query_sample (torch.Tensor): Query sample tensor [num_features]
            historical_samples (torch.Tensor): Historical samples tensor [num_samples, num_features]
            historical_outputs (torch.Tensor): Historical outputs tensor [num_samples]
            bmus_idx_map (dict[tuple[int, int], list[int]]): BMU to data indices mapping
            min_buffer_threshold (int): Minimum buffer threshold
            return_indices (bool): If True, also return the indices of collected samples

        Returns:
            If return_indices is False: (historical_data_buffer, historical_output_buffer)
            If return_indices is True: (historical_data_buffer, historical_output_buffer, indices_tensor)
        """
        query_sample = query_sample.to(self.device)
        bmu_pos = self.identify_bmus(query_sample)
        bmu_row, bmu_col = int(bmu_pos[0].item()), int(bmu_pos[1].item())
        bmu_tuple = (bmu_row, bmu_col)
        if self.topology == "rectangular":
            offsets = self._neighbor_offsets
        else:
            row_type = "even" if bmu_row % 2 == 0 else "odd"
            offsets = self._neighbor_offsets[row_type]

        collected_sample_indices = list(bmus_idx_map.get(bmu_tuple, []))
        visited_neurons = {bmu_tuple}
        for dx, dy in offsets:
            nr, nc = bmu_row + dx, bmu_col + dy
            if self.pbc:
                nr, nc = nr % self.x, nc % self.y
            elif not (0 <= nr < self.x and 0 <= nc < self.y):
                continue
            pos = (nr, nc)
            if pos not in visited_neurons and pos in bmus_idx_map:
                collected_sample_indices.extend(bmus_idx_map[pos])
                visited_neurons.add(pos)

        # If we need more samples, use distance-based collection
        if len(collected_sample_indices) <= min_buffer_threshold:
            bmu_weights = self.weights[bmu_row, bmu_col]
            distances = self._calculate_distances_to_neurons(bmu_weights)

            # Identify all unvisited neurons with samples, sorted by distance
            candidate_neurons = []
            for (r, c), samples in bmus_idx_map.items():
                if (r, c) not in visited_neurons and samples:
                    dist = distances[r, c].item()
                    candidate_neurons.append((dist, r, c))
            candidate_neurons.sort(key=lambda x: x[0])

            # Collect samples from candidate unvisitedneurons
            for _, r, c in candidate_neurons:
                collected_sample_indices.extend(bmus_idx_map[(r, c)])
                visited_neurons.add((r, c))
                if len(collected_sample_indices) > min_buffer_threshold:
                    break

        # Build buffers
        indices_tensor = torch.tensor(
            collected_sample_indices,
            device=historical_samples.device,
            dtype=torch.long,
        )
        historical_data_buffer = historical_samples[indices_tensor]
        historical_output_buffer = historical_outputs[indices_tensor].view(-1, 1)
        if return_indices:
            return historical_data_buffer, historical_output_buffer, indices_tensor
        return historical_data_buffer, historical_output_buffer

    def build_map(
        self,
        map_type: str,
        data: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        bmus_data_map: Optional[dict[tuple[int, int], list[int]]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Unified method to build various types of maps.

        Args:
            map_type (str): Type of map to build. Options:
                - "hit": Hit map showing neuron activation frequencies
                - "distance": Distance map showing neuron-to-neighbor distances
                - "bmus_data": Mapping of BMUs to their corresponding data points
                - "metric": Metric map based on target values (requires target)
                - "score": Score map combining standard error with distribution penalty (requires target)
                - "rank": Rank map based on neuron standard deviations (requires target)
                - "classification": Classification map with most frequent labels (requires target)
            data (Optional[torch.Tensor]): Input data tensor [batch_size, num_features].
                Required if bmus_data_map is not provided.
            target (Optional[torch.Tensor]): Target values/labels (required for some map types)
            bmus_data_map (Optional[dict[tuple[int, int], list[int]]]): Pre-computed BMU to data indices mapping.
                If provided, avoids recomputing BMUs for dependent maps.
            **kwargs: Additional arguments specific to each map type:
                - batch_size (int): Batch processing size (default: 1024)
                - distance_metric (str): Distance function for distance maps
                - neighborhood_order (int): Neighborhood order for distance/classification maps
                - scaling (str): 'sum' or 'mean' for distance maps
                - reduction_parameter (str): 'mean' or 'std' for metric maps
                - return_indices (bool): Return indices instead of data for bmus_data maps

        Returns:
            torch.Tensor or Dict: Map result (type depends on map_type)

        Raises:
            ValueError: If invalid map_type is specified
            ValueError: If target is required but not provided
            ValueError: If neither data nor bmus_data_map is provided
        """
        bmus_dependent_maps = {"metric", "score", "rank", "classification"}
        data_dependent_maps = {"hit", "bmus_data"}
        target_required_maps = {"metric", "score", "rank", "classification"}
        if map_type not in MAP_FUNCTIONS:
            available_types = ", ".join(MAP_FUNCTIONS.keys())
            raise ValueError(
                f"Invalid map_type '{map_type}'. Available types: {available_types}"
            )
        if map_type in target_required_maps and target is None:
            raise ValueError(f"Map type '{map_type}' requires target parameter")
        if map_type in data_dependent_maps and data is None:
            raise ValueError(f"Map type '{map_type}' requires data parameter")
        if map_type in bmus_dependent_maps:
            if bmus_data_map is None and data is None:
                raise ValueError(
                    f"Map type '{map_type}' requires either data or bmus_data_map parameter"
                )

        map_function = MAP_FUNCTIONS[map_type]
        if map_type in data_dependent_maps:
            if map_type in target_required_maps:
                return map_function(self, data, target, **kwargs)
            else:
                return map_function(self, data, **kwargs)

        elif map_type in bmus_dependent_maps:
            if bmus_data_map is None:
                bmus_data_map = MAP_FUNCTIONS["bmus_data"](
                    self, data, return_indices=True
                )
            if map_type == "score":
                if "total_samples" not in kwargs:
                    if data is not None:
                        total_samples = len(data)
                    else:
                        total_samples = sum(
                            len(indices) for indices in bmus_data_map.values()
                        )
                    kwargs["total_samples"] = total_samples
                return map_function(self, bmus_data_map, target, **kwargs)
            else:
                return map_function(self, bmus_data_map, target, **kwargs)

        elif map_type == "distance":
            return map_function(self, **kwargs)

        else:
            raise ValueError(f"Unknown map type handling for: {map_type}")

    def build_multiple_maps(
        self,
        map_configs: list[dict[str, Any]],
        data: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        batch_size: int = 1024,
    ) -> dict[str, torch.Tensor]:
        """Efficiently build multiple maps by reusing BMUs computation.

        Args:
            map_configs (list[dict]): List of map configurations
            data (torch.Tensor): Input data tensor
            target (Optional[torch.Tensor]): Target values (if needed by any map)
            batch_size (int): Batch size for BMUs computation

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping map names to their results

        Example:
            configs = [
                {'type': 'hit'},
                {'type': 'metric', 'kwargs': {'reduction_parameter': 'std'}},
                {'type': 'rank'},
                {'type': 'classification', 'kwargs': {'neighborhood_order': 2}}
            ]
            results = som.build_multiple_maps(configs, data, target)
        """
        data_dependent_maps = {"hit", "bmus_data"}
        bmus_dependent_maps = {"metric", "score", "rank", "classification"}
        need_bmus_map = any(
            config["type"] in bmus_dependent_maps for config in map_configs
        )
        results = {}
        bmus_data_map = None
        if need_bmus_map:
            bmus_data_map = self.build_map(
                "bmus_data", data=data, return_indices=True, batch_size=batch_size
            )

        for config in map_configs:
            map_type = config["type"]
            map_kwargs = config.get("kwargs", {})
            # Essential to separate maps with the same method but different parameters: metric_std vs metric_mean
            if map_kwargs:
                key = f"{map_type}_{hash(str(sorted(map_kwargs.items())))}"
            else:
                key = map_type
            if map_type in data_dependent_maps:
                results[key] = self.build_map(
                    map_type, data=data, target=target, **map_kwargs
                )
            elif map_type in bmus_dependent_maps:
                results[key] = self.build_map(
                    map_type, target=target, bmus_data_map=bmus_data_map, **map_kwargs
                )
            else:
                raise ValueError(f"Unknown map type: {map_type}")

        return results

    def cluster(
        self,
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        feature_space: str = "weights",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Cluster SOM neurons using various clustering algorithms.

        Args:
            method (str): Clustering method. Options: "kmeans", "gmm", "hdbscan"
            n_clusters (Optional[int]): Number of clusters. If None, uses automatic selection
            feature_space (str): Feature space for clustering. Options:
                - "weights": Cluster based on neuron weight vectors
                - "positions": Cluster based on 2D neuron coordinates
                - "combined": Use both weights and positions
            **kwargs: Additional arguments for clustering algorithms

        Returns:
            dict[str, Any]: Clustering results containing:
                - labels: Cluster assignments for neurons [n_neurons]
                - centers: Cluster centers [n_clusters, n_features]
                - n_clusters: Number of clusters found
                - method: Clustering method used
                - metrics: Dictionary of clustering quality metrics
                - feature_space: Feature space used for clustering
                - original_data: Features used for clustering

        Raises:
            ValueError: If invalid method or feature_space is specified
        """
        if method not in ["kmeans", "gmm", "hdbscan"]:
            raise ValueError(f"Unsupported clustering method: {method}")
        if feature_space not in ["weights", "positions", "combined"]:
            raise ValueError(f"Unsupported feature space: {feature_space}")

        data = extract_clustering_features(som=self, feature_space=feature_space)
        cluster_result = cluster_data(
            data=data, method=method, n_clusters=n_clusters, **kwargs
        )

        metrics = calculate_clustering_metrics(data, cluster_result["labels"], som=self)
        cluster_result["metrics"] = metrics
        cluster_result["feature_space"] = feature_space
        cluster_result["original_data"] = data
        return cluster_result
