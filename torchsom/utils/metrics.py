"""Utility functions for metrics."""

import warnings
from typing import TYPE_CHECKING, Callable

import torch
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from torchsom.utils.hexagonal_coordinates import (
    hexagonal_distance_axial,
    offset_to_axial_coords,
)

if TYPE_CHECKING:
    from torchsom.core.base_som import BaseSOM


def calculate_quantization_error(
    data: torch.Tensor,
    weights: torch.Tensor,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    """Calculate quantization error for a SOM.

    Args:
        data (torch.Tensor): Input data tensor [batch_size, num_features] or [num_features]
        weights (torch.Tensor): SOM weights [x, y, num_features]
        distance_fn (Callable): Function to compute distances between data and weights

    Returns:
        float: Average quantization error value
    """
    device = weights.device
    data = data.to(device)
    if data.dim() == 1:
        data = data.unsqueeze(0)
    # Calculate distances between each data point and all neurons: [batch_size, x, y]
    distances = distance_fn(data, weights)
    # Flatten distances: [batch_size, x*y]
    distances_flat = distances.view(data.shape[0], -1)
    # Find minimum distances efficiently: [batch_size]
    min_distances = torch.min(distances_flat, dim=1)[0]
    return min_distances.mean().item()


def calculate_topographic_error(
    data: torch.Tensor,
    weights: torch.Tensor,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    topology: str = "rectangular",
) -> float:
    """Calculate topographic error for a SOM.

    Args:
        data (torch.Tensor): Input data tensor [batch_size, num_features] or [num_features]
        weights (torch.Tensor): SOM weights [x, y, num_features]
        distance_fn (Callable): Function to compute distances between data and weights
        topology (str, optional): Grid configuration. Defaults to "rectangular".

    Returns:
        float: Topographic error ratio
    """
    device = weights.device
    data = data.to(device)
    if data.dim() == 1:
        data = data.unsqueeze(0)

    x_dim, y_dim = weights.shape[0], weights.shape[1]
    if x_dim * y_dim == 1:
        warnings.warn(
            "The topographic error is not defined for a 1-by-1 map.",
            stacklevel=2,
        )
        return float("nan")

    batch_size = data.shape[0]
    # Calculate distances between each data point and all neurons: [batch_size, x, y]
    distances = distance_fn(data, weights)
    # Flatten distances: [batch_size, x*y]
    distances_flat = distances.view(batch_size, -1)
    # Get top 2 BMU indices efficiently: [batch_size, 2]
    _, indices = torch.topk(distances_flat, k=2, largest=False, dim=1)

    # Convert flat indices to 2D coordinates
    bmu1_row, bmu1_col = (
        torch.div(indices[:, 0], y_dim, rounding_mode="floor"),
        indices[:, 0] % y_dim,
    )
    bmu2_row, bmu2_col = (
        torch.div(indices[:, 1], y_dim, rounding_mode="floor"),
        indices[:, 1] % y_dim,
    )
    if topology == "hexagonal":
        error_count = 0
        for i in range(batch_size):
            q1, r1 = offset_to_axial_coords(bmu1_row[i].item(), bmu1_col[i].item())
            q2, r2 = offset_to_axial_coords(bmu2_row[i].item(), bmu2_col[i].item())
            hex_distance = hexagonal_distance_axial(q1, r1, q2, r2)
            if hex_distance > 1:
                error_count += 1
        return error_count / batch_size
    else:
        dx = (bmu2_row - bmu1_row).float()
        dy = (bmu2_col - bmu1_col).float()
        distances = torch.sqrt(dx**2 + dy**2)
        threshold = 1.0
        return (distances > threshold).float().mean().item()


def calculate_silhouette_score(
    data: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Calculate silhouette score for clustering results using scikit-learn.

    Args:
        data (torch.Tensor): Input data [n_samples, n_features]
        labels (torch.Tensor): Cluster labels [n_samples]

    Returns:
        float: Silhouette score (-1 to 1, higher is better)
    """
    if data.shape[0] != labels.shape[0]:
        raise ValueError("Data and labels must have the same number of samples")

    # Convert to numpy for sklearn
    data_np = data.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Remove noise points (label -1) for valid silhouette calculation
    noise_mask = labels_np == -1
    if noise_mask.sum() == len(labels_np):
        return 0.0  # All points are noise

    if noise_mask.sum() > 0:
        valid_mask = ~noise_mask
        data_clean = data_np[valid_mask]
        labels_clean = labels_np[valid_mask]
    else:
        data_clean = data_np
        labels_clean = labels_np

    # Check if we have enough clusters for silhouette score
    n_unique = len(set(labels_clean))
    if n_unique <= 1 or len(data_clean) <= 1:
        return 0.0

    try:
        return float(silhouette_score(data_clean, labels_clean))
    except Exception:
        return 0.0


def calculate_davies_bouldin_score(
    data: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Calculate Davies-Bouldin index using scikit-learn.

    Args:
        data (torch.Tensor): Input data [n_samples, n_features]
        labels (torch.Tensor): Cluster labels [n_samples]

    Returns:
        float: Davies-Bouldin index (lower is better, >= 0)
    """
    if data.shape[0] != labels.shape[0]:
        raise ValueError("Data and labels must have the same number of samples")

    # Convert to numpy for sklearn
    data_np = data.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Remove noise points (label -1)
    noise_mask = labels_np == -1
    if noise_mask.sum() == len(labels_np):
        return float("inf")  # All points are noise

    if noise_mask.sum() > 0:
        valid_mask = ~noise_mask
        data_clean = data_np[valid_mask]
        labels_clean = labels_np[valid_mask]
    else:
        data_clean = data_np
        labels_clean = labels_np

    # Check if we have enough clusters
    n_unique = len(set(labels_clean))
    if n_unique <= 1:
        return 0.0

    try:
        return float(davies_bouldin_score(data_clean, labels_clean))
    except Exception:
        return float("inf")


def calculate_calinski_harabasz_score(
    data: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Calculate Calinski-Harabasz index using scikit-learn.

    Args:
        data (torch.Tensor): Input data [n_samples, n_features]
        labels (torch.Tensor): Cluster labels [n_samples]

    Returns:
        float: Calinski-Harabasz index (higher is better, >= 0)
    """
    if data.shape[0] != labels.shape[0]:
        raise ValueError("Data and labels must have the same number of samples")

    # Convert to numpy for sklearn
    data_np = data.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Remove noise points (label -1)
    noise_mask = labels_np == -1
    if noise_mask.sum() == len(labels_np):
        return 0.0  # All points are noise

    if noise_mask.sum() > 0:
        valid_mask = ~noise_mask
        data_clean = data_np[valid_mask]
        labels_clean = labels_np[valid_mask]
    else:
        data_clean = data_np
        labels_clean = labels_np

    # Check if we have enough clusters and samples
    n_unique = len(set(labels_clean))
    if n_unique <= 1 or len(data_clean) <= n_unique:
        return 0.0

    try:
        return float(calinski_harabasz_score(data_clean, labels_clean))
    except Exception:
        return 0.0


def calculate_topological_clustering_quality(
    som: "BaseSOM",
    labels: torch.Tensor,
) -> float:
    """Calculate how well clusters respect SOM topological structure.

    This metric measures the spatial coherence of clusters on the SOM grid.
    Higher values indicate that clusters are more spatially compact on the grid.

    Args:
        som (BaseSOM): Trained SOM instance
        labels (torch.Tensor): Cluster labels for neurons [n_neurons]

    Returns:
        float: Topological clustering quality (0 to 1, higher is better)
    """
    if labels.shape[0] != som.x * som.y:
        raise ValueError("Labels must have one entry per neuron")

    # Reshape labels to grid
    labels_grid = labels.view(som.x, som.y)

    # Remove noise points (label -1)
    unique_labels = torch.unique(labels)
    valid_labels = unique_labels[unique_labels != -1]

    if len(valid_labels) <= 1:
        return 1.0  # Perfect if only one cluster or no valid clusters

    total_coherence = 0.0
    total_neurons = 0

    for label in valid_labels:
        # Find all neurons with this label
        cluster_mask = labels_grid == label
        cluster_positions = torch.where(cluster_mask)

        if len(cluster_positions[0]) <= 1:
            continue  # Skip single-neuron clusters

        # Calculate pairwise distances between neurons in this cluster
        cluster_coords = torch.stack(
            [cluster_positions[0], cluster_positions[1]], dim=1
        ).float()

        if som.topology == "rectangular":
            # Euclidean distance for rectangular topology
            pairwise_distances = torch.cdist(cluster_coords, cluster_coords)
        else:
            # Hexagonal distance for hexagonal topology
            n_cluster_neurons = len(cluster_coords)
            pairwise_distances = torch.zeros(n_cluster_neurons, n_cluster_neurons)

            for i in range(n_cluster_neurons):
                for j in range(i + 1, n_cluster_neurons):
                    row1, col1 = int(cluster_coords[i, 0]), int(cluster_coords[i, 1])
                    row2, col2 = int(cluster_coords[j, 0]), int(cluster_coords[j, 1])

                    q1, r1 = offset_to_axial_coords(row1, col1)
                    q2, r2 = offset_to_axial_coords(row2, col2)
                    hex_dist = hexagonal_distance_axial(q1, r1, q2, r2)

                    pairwise_distances[i, j] = hex_dist
                    pairwise_distances[j, i] = hex_dist

        # Calculate average distance within cluster
        n_pairs = len(cluster_coords)
        if n_pairs > 1:
            mask = ~torch.eye(n_pairs, dtype=torch.bool)
            avg_distance = pairwise_distances[mask].mean().item()

            # Normalize by maximum possible distance on grid
            max_distance = max(som.x, som.y)
            normalized_distance = avg_distance / max_distance

            # Convert to coherence (inverse of distance)
            coherence = 1.0 / (1.0 + normalized_distance)

            total_coherence += coherence * len(cluster_coords)
            total_neurons += len(cluster_coords)

    if total_neurons == 0:
        return 1.0

    return total_coherence / total_neurons


def calculate_clustering_metrics(
    data: torch.Tensor,
    labels: torch.Tensor,
    som: "BaseSOM" = None,
) -> dict[str, float]:
    """Calculate comprehensive clustering quality metrics.

    Args:
        data (torch.Tensor): Input data [n_samples, n_features]
        labels (torch.Tensor): Cluster labels [n_samples]
        som (Optional[BaseSOM]): SOM instance for topological metrics

    Returns:
        dict[str, float]: Dictionary of clustering quality metrics
    """
    metrics = {}

    # Standard clustering metrics using sklearn
    metrics["silhouette_score"] = calculate_silhouette_score(data, labels)
    metrics["davies_bouldin_score"] = calculate_davies_bouldin_score(data, labels)
    metrics["calinski_harabasz_score"] = calculate_calinski_harabasz_score(data, labels)

    # Basic statistics
    unique_labels = torch.unique(labels)
    metrics["n_clusters"] = len(unique_labels[unique_labels != -1])
    metrics["n_noise_points"] = (labels == -1).sum().item()
    metrics["noise_ratio"] = metrics["n_noise_points"] / len(labels)

    # SOM-specific metrics if SOM is provided
    if som is not None:
        try:
            metrics["topological_quality"] = calculate_topological_clustering_quality(
                som, labels
            )
        except Exception as e:
            warnings.warn(f"Could not calculate topological quality: {e}", stacklevel=2)
            metrics["topological_quality"] = float("nan")

    return metrics
