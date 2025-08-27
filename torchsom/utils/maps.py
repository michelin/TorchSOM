"""Optimized GPU-accelerated map building functions for TorchSOM."""

import random
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Optional

import torch

from torchsom.utils.distances import PAIRWISE_DISTANCE_FUNCTIONS
from torchsom.utils.topology import get_all_neighbors_up_to_order

if TYPE_CHECKING:
    from torchsom.core.base_som import BaseSOM


def build_bmus_data_map(
    som_instance: "BaseSOM",
    data: torch.Tensor,
    return_indices: bool = False,
    batch_size: int = 1024,
) -> dict[tuple[int, int], Any]:
    """Build mapping of BMUs to their corresponding data points.

    Args:
        som_instance (BaseSOM): SOM instance
        data (torch.Tensor): Input data tensor
        return_indices (bool): Return indices instead of data points
        batch_size (int): Batch processing size

    Returns:
        dict[tuple[int, int], Any]: Dictionary mapping BMU coordinates to data/indices
    """
    bmus_data_map = defaultdict(list)
    num_samples = data.shape[0]

    for batch_idx in range(0, num_samples, batch_size):
        batch_data = data[batch_idx : batch_idx + batch_size].to(som_instance.device)
        batch_bmus = som_instance.identify_bmus(batch_data)
        if batch_bmus.dim() == 1:
            batch_bmus = batch_bmus.unsqueeze(0)
        for i, (row, col) in enumerate(batch_bmus):
            bmu_pos = (int(row.item()), int(col.item()))
            global_idx = batch_idx + i
            if return_indices:
                bmus_data_map[bmu_pos].append(global_idx)
            else:
                bmus_data_map[bmu_pos].append(batch_data[i])

    # Convert lists to tensors if returning data points
    if not return_indices:
        for bmu in bmus_data_map:
            if bmus_data_map[bmu]:
                bmus_data_map[bmu] = torch.stack(bmus_data_map[bmu])

    return bmus_data_map


def build_hit_map(
    som_instance: "BaseSOM",
    data: torch.Tensor,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Build hit map showing neuron activation frequencies.

    Args:
        som_instance (BaseSOM): SOM instance
        data (torch.Tensor): Input data tensor [batch_size, num_features]
        batch_size (int): Batch processing size

    Returns:
        torch.Tensor: Hit map tensor [x, y]
    """
    hit_map = torch.zeros((som_instance.x, som_instance.y), device=som_instance.device)
    num_samples = data.shape[0]

    for batch_idx in range(0, num_samples, batch_size):
        batch_data = data[batch_idx : batch_idx + batch_size].to(som_instance.device)
        batch_bmus = som_instance.identify_bmus(batch_data)
        if batch_bmus.dim() == 1:
            batch_bmus = batch_bmus.unsqueeze(0)
        flat_indices = batch_bmus[:, 0] * som_instance.y + batch_bmus[:, 1]
        hit_map.view(-1).scatter_add_(
            0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32)
        )

    return hit_map


def build_distance_map(
    som_instance: "BaseSOM",
    distance_metric: Optional[str] = None,
    neighborhood_order: Optional[int] = None,
    scaling: str = "sum",
) -> torch.Tensor:
    """Build distance map showing neuron-to-neighbor distances.

    Args:
        som_instance (BaseSOM): SOM instance
        distance_metric (str): Distance function name
        neighborhood_order (int): Neighbor order to consider
        scaling (str): 'sum' or 'mean' aggregation

    Returns:
        torch.Tensor: Distance map tensor [x, y]
    """
    if scaling not in ["sum", "mean"]:
        raise ValueError(
            f'scaling should be either "sum" or "mean" ({scaling} is not valid)'
        )
    if neighborhood_order is None:
        neighborhood_order = som_instance.neighborhood_order
    if distance_metric is None:
        distance_metric = som_instance.distance_fn_name

    if distance_metric not in PAIRWISE_DISTANCE_FUNCTIONS:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    pairwise_distance_fn = PAIRWISE_DISTANCE_FUNCTIONS[distance_metric]

    if neighborhood_order == som_instance.neighborhood_order:
        all_offsets = som_instance._neighbor_offsets
    else:
        all_offsets = get_all_neighbors_up_to_order(
            topology=som_instance.topology,
            max_order=neighborhood_order,
        )
    device = som_instance.device
    weights_flat = som_instance.weights.view(-1, som_instance.num_features)
    grid_i, grid_j = torch.meshgrid(
        torch.arange(som_instance.x, device=device),
        torch.arange(som_instance.y, device=device),
        indexing="ij",
    )
    distance_map = torch.zeros((som_instance.x, som_instance.y), device=device)
    counts = torch.zeros((som_instance.x, som_instance.y), device=device)

    if som_instance.topology == "hexagonal":
        for row in range(som_instance.x):
            row_offsets = all_offsets["even"] if row % 2 == 0 else all_offsets["odd"]
            for row_offset, col_offset in row_offsets:
                neighbor_row = row + row_offset
                neighbor_col_indices = (
                    torch.arange(som_instance.y, device=device) + col_offset
                )
                valid_mask = (
                    (neighbor_row >= 0)
                    & (neighbor_row < som_instance.x)
                    & (neighbor_col_indices >= 0)
                    & (neighbor_col_indices < som_instance.y)
                )
                if valid_mask.any():
                    # Get flattened indices for current and neighbor positions
                    current_indices = row * som_instance.y + torch.arange(
                        som_instance.y, device=device
                    )
                    neighbor_indices = (
                        neighbor_row * som_instance.y + neighbor_col_indices
                    )
                    valid_current = current_indices[valid_mask]
                    valid_neighbors = neighbor_indices[valid_mask]
                    # Compute distances
                    current_weights = weights_flat[valid_current]
                    neighbor_weights = weights_flat[valid_neighbors]
                    distances = pairwise_distance_fn(current_weights, neighbor_weights)
                    # Add to distance map
                    valid_positions = torch.stack(
                        [
                            torch.full_like(valid_current, row),
                            torch.arange(som_instance.y, device=device)[valid_mask],
                        ],
                        dim=1,
                    )
                    distance_map[
                        valid_positions[:, 0], valid_positions[:, 1]
                    ] += distances
                    counts[valid_positions[:, 0], valid_positions[:, 1]] += 1

    else:
        for row_offset, col_offset in all_offsets:
            neighbor_rows = grid_i + row_offset
            neighbor_cols = grid_j + col_offset
            valid_mask = (
                (neighbor_rows >= 0)
                & (neighbor_rows < som_instance.x)
                & (neighbor_cols >= 0)
                & (neighbor_cols < som_instance.y)
            )
            if valid_mask.any():
                # Get flattened indices for current and neighbor positions
                current_flat = (grid_i * som_instance.y + grid_j)[valid_mask]
                neighbor_flat = (neighbor_rows * som_instance.y + neighbor_cols)[
                    valid_mask
                ]
                # Compute distances
                current_weights = weights_flat[current_flat]
                neighbor_weights = weights_flat[neighbor_flat]
                distances = pairwise_distance_fn(current_weights, neighbor_weights)
                # Add to distance map using advanced indexing
                valid_positions = torch.nonzero(valid_mask, as_tuple=False)
                distance_map[valid_positions[:, 0], valid_positions[:, 1]] += distances
                counts[valid_positions[:, 0], valid_positions[:, 1]] += 1

    if scaling == "mean":
        distance_map = torch.where(counts > 0, distance_map / counts, distance_map)
    max_distance = distance_map.max()
    if max_distance > 0:
        distance_map = distance_map / max_distance
    return distance_map


def build_metric_map(
    som_instance: "BaseSOM",
    bmus_data_map: dict[tuple[int, int], Any],
    target: torch.Tensor,
    reduction_parameter: str,
) -> torch.Tensor:
    """Build metric map based on target values using pre-computed BMUs map.

    Args:
        som_instance (BaseSOM): SOM instance
        bmus_data_map (dict[tuple[int, int], Any]): Pre-computed BMU to data indices mapping
        target (torch.Tensor): Target values tensor
        reduction_parameter (str): 'mean' or 'std'

    Returns:
        torch.Tensor: Metric map tensor [x, y]
    """
    device = som_instance.device
    target = target.to(device)
    epsilon = torch.tensor(1e-8, device=device)
    metric_map = torch.full(
        (som_instance.x, som_instance.y), float("nan"), device=device
    )

    for bmu_pos, sample_indices in bmus_data_map.items():
        if len(sample_indices) > 0:
            if isinstance(sample_indices, list):
                indices_tensor = torch.tensor(sample_indices, device=device)
            else:
                indices_tensor = sample_indices.to(device)

            target_values = target[indices_tensor].float()
            if reduction_parameter == "mean":
                metric_map[bmu_pos] = target_values.mean()
            elif reduction_parameter == "std":
                if len(sample_indices) > 1:
                    metric_map[bmu_pos] = target_values.std(unbiased=True)
                else:
                    metric_map[bmu_pos] = epsilon

    return metric_map


def build_score_map(
    som_instance: "BaseSOM",
    bmus_data_map: dict[tuple[int, int], Any],
    target: torch.Tensor,
    total_samples: int,
) -> torch.Tensor:
    """Build score map combining standard error with distribution penalty using pre-computed BMUs map.

    Args:
        som_instance (BaseSOM): SOM instance
        bmus_data_map (dict[tuple[int, int], Any]): Pre-computed BMU to data indices mapping
        target (torch.Tensor): Target values tensor
        total_samples (int): Total number of data samples

    Returns:
        torch.Tensor: Score map tensor [x, y]
    """
    device = som_instance.device
    target = target.to(device)
    epsilon = torch.tensor(1e-8, device=device)
    total_samples_tensor = torch.tensor(
        total_samples, dtype=torch.float32, device=device
    )
    score_map = torch.full(
        (som_instance.x, som_instance.y), float("nan"), device=device
    )

    for bmu_pos, sample_indices in bmus_data_map.items():
        if len(sample_indices) > 0:

            # Multiple samples in neuron
            if len(sample_indices) > 1:
                if isinstance(sample_indices, list):
                    indices_tensor = torch.tensor(sample_indices, device=device)
                else:
                    indices_tensor = sample_indices.to(device)
                target_values = target[indices_tensor].float()
                std = target_values.std(unbiased=True)
                n_samples = torch.tensor(
                    len(sample_indices), dtype=torch.float32, device=device
                )
                neuron_score = (std / torch.sqrt(n_samples)) * torch.log(
                    total_samples_tensor / n_samples
                )
                score_map[bmu_pos] = torch.max(neuron_score, epsilon)

            # Single sample in neuron
            else:
                score_map[bmu_pos] = epsilon

    return score_map


def build_rank_map(
    som_instance: "BaseSOM",
    bmus_data_map: dict[tuple[int, int], Any],
    target: torch.Tensor,
) -> torch.Tensor:
    """Build rank map based on neuron standard deviations using pre-computed BMUs map.

    Args:
        som_instance (BaseSOM): SOM instance
        bmus_data_map (dict[tuple[int, int], Any]): Pre-computed BMU to data indices mapping
        target (torch.Tensor): Target values tensor

    Returns:
        torch.Tensor: Rank map tensor [x, y]
    """
    device = som_instance.device
    target = target.to(device)
    active_positions = []
    std_values = []

    # Compute list of std values for each neuron
    for bmu_pos, sample_indices in bmus_data_map.items():
        if len(sample_indices) > 0:
            active_positions.append(bmu_pos)

            # Multiple samples in neuron
            if len(sample_indices) > 1:
                if isinstance(sample_indices, list):
                    indices_tensor = torch.tensor(sample_indices, device=device)
                else:
                    indices_tensor = sample_indices.to(device)
                target_values = target[indices_tensor].float()
                std_val = target_values.std(unbiased=True)

            # Single sample in neuron
            else:
                std_val = torch.tensor(0.0, device=device)
            std_values.append(std_val)

    rank_map = torch.full((som_instance.x, som_instance.y), float("nan"), device=device)
    if std_values:
        std_tensor = torch.stack(std_values)
        ranks = torch.argsort(std_tensor, descending=True).argsort() + 1
        for i, pos in enumerate(active_positions):
            rank_map[pos] = ranks[i].float()

    return rank_map


def build_classification_map(
    som_instance: "BaseSOM",
    bmus_data_map: dict[tuple[int, int], Any],
    target: torch.Tensor,
    neighborhood_order: int = 1,
) -> torch.Tensor:
    """Build classification map with most frequent labels per neuron using pre-computed BMUs map.

    Args:
        som_instance (BaseSOM): SOM instance
        bmus_data_map (dict[tuple[int, int], Any]): Pre-computed BMU to data indices mapping
        target (torch.Tensor): Target labels tensor
        neighborhood_order (int): Neighborhood order for tie-breaking

    Returns:
        torch.Tensor: Classification map tensor [x, y]
    """
    device = som_instance.device
    target = target.to(device)
    classification_map = torch.full(
        (som_instance.x, som_instance.y), float("nan"), device=device
    )

    if neighborhood_order == som_instance.neighborhood_order:
        neighborhood_offsets = som_instance._neighbor_offsets
    else:
        neighborhood_offsets = get_all_neighbors_up_to_order(
            topology=som_instance.topology,
            max_order=neighborhood_order,
        )

    for bmu_pos, sample_indices in bmus_data_map.items():
        if len(sample_indices) > 0:
            if isinstance(sample_indices, list):
                indices_tensor = torch.tensor(sample_indices, device=device)
            else:
                indices_tensor = sample_indices.to(device)

            neuron_labels = target[indices_tensor]
            unique_labels, counts = torch.unique(neuron_labels, return_counts=True)
            max_count = counts.max()
            top_labels_mask = counts == max_count
            top_labels = unique_labels[top_labels_mask]

            if len(top_labels) == 1:
                classification_map[bmu_pos] = top_labels[0].float()
            else:
                neighbor_labels = []
                row, col = bmu_pos
                if som_instance.topology == "hexagonal":
                    row_offsets = (
                        neighborhood_offsets["even"]
                        if row % 2 == 0
                        else neighborhood_offsets["odd"]
                    )
                else:
                    row_offsets = neighborhood_offsets

                for dx, dy in row_offsets:
                    neighbor_row, neighbor_col = row + dx, col + dy
                    neighbor_pos = (neighbor_row, neighbor_col)

                    if (
                        0 <= neighbor_row < som_instance.x
                        and 0 <= neighbor_col < som_instance.y
                        and neighbor_pos in bmus_data_map
                    ):
                        neighbor_indices = bmus_data_map[neighbor_pos]
                        if isinstance(neighbor_indices, list):
                            neighbor_tensor = torch.tensor(
                                neighbor_indices, device=device
                            )
                        else:
                            neighbor_tensor = neighbor_indices.to(device)
                        neighbor_labels.extend(target[neighbor_tensor].tolist())

                # Use neighborhood information to break ties
                if neighbor_labels:
                    expanded_counts = Counter(neighbor_labels)
                    max_neighbor_count = max(expanded_counts.values())
                    top_neighbor_labels = [
                        label
                        for label, count in expanded_counts.items()
                        if count == max_neighbor_count
                    ]
                    chosen_label = random.choice(top_neighbor_labels)
                else:
                    chosen_label = random.choice(top_labels.tolist())

                classification_map[bmu_pos] = torch.tensor(
                    chosen_label, dtype=classification_map.dtype, device=device
                )

    return classification_map


MAP_FUNCTIONS = {
    "hit": build_hit_map,
    "distance": build_distance_map,
    "bmus_data": build_bmus_data_map,
    "metric": build_metric_map,
    "score": build_score_map,
    "rank": build_rank_map,
    "classification": build_classification_map,
}
