"""Utility functions for distances."""

import torch
import torch.nn.functional as F


def _euclidean_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute Euclidean (L2) distance between input and weights.

    Args:
        data (torch.Tensor): [batch_size, num_features] tensor
        weights (torch.Tensor): [x, y, num_features] tensor

    Returns:
        torch.Tensor: tensor of Euclidean distances between input data and weights [batch_size, x, y]
    """
    batch_size = data.shape[0]
    x, y, n_features = weights.shape
    # Flatten weights for cdist: [x*y, n_features]
    weights_flat = weights.view(-1, n_features)
    # Calculate distances: [batch_size, x*y]
    distances = torch.cdist(data, weights_flat, p=2)
    return distances.view(batch_size, x, y)


def _cosine_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine distance between input and weights.

    Args:
        data (torch.Tensor): [batch_size, num_features] tensor
        weights (torch.Tensor): [x, y, num_features] tensor

    Returns:
        torch.Tensor: tensor of cosine distances between input data and weights [batch_size, x, y]
    """
    batch_size, n_features = data.shape
    x, y, _ = weights.shape
    # Flatten weights: [x*y, n_features]
    weights_flat = weights.view(-1, n_features)
    # Normalize data and weights: [batch_size, n_features] and [x*y, n_features]
    data_norm = F.normalize(data, dim=-1)
    weights_norm = F.normalize(weights_flat, dim=-1)
    # Cosine similarity: [batch_size, x*y]
    cos_sim = torch.mm(data_norm, weights_norm.T)
    # Convert to distance: [batch_size, x*y]
    cos_dist = torch.clamp(1 - cos_sim, 0.0, 1.0)
    return cos_dist.view(batch_size, x, y)


def _manhattan_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute Manhattan (L1) distance between input and weights.

    Args:
        data (torch.Tensor): [batch_size, num_features] tensor
        weights (torch.Tensor): [x, y, num_features] tensor

    Returns:
        torch.Tensor: tensor of Manhattan distances between input data and weights [batch_size, x, y]
    """
    batch_size, n_features = data.shape
    x, y, _ = weights.shape
    # Flatten weights: [x*y, n_features]
    weights_flat = weights.view(-1, n_features)
    # Calculate distances: [batch_size, x*y]
    distances = torch.cdist(data, weights_flat, p=1)
    return distances.view(batch_size, x, y)


def _chebyshev_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute Chebyshev (L∞) distance between input and weights.

    Can't use cdist as p needs to be finite, so we use max reduction.

    Args:
        data (torch.Tensor): [batch_size, num_features] tensor
        weights (torch.Tensor): [x, y, num_features] tensor

    Returns:
        torch.Tensor: tensor of Chebyshev distances between input data and weights [batch_size, x, y]
    """
    batch_size, n_features = data.shape
    x, y, _ = weights.shape
    # Flatten weights: [x*y, n_features]
    weights_flat = weights.view(-1, n_features)
    # Pairwise absolute diffs: [batch_size, x*y, n_features]
    diffs = torch.abs(data.unsqueeze(1) - weights_flat.unsqueeze(0))
    # Max across features, resulting in L∞ distance: [batch_size, x*y]
    distances = diffs.max(dim=-1).values
    return distances.view(batch_size, x, y)


DISTANCE_FUNCTIONS = {
    "euclidean": _euclidean_distance,
    "cosine": _cosine_distance,
    "manhattan": _manhattan_distance,
    "chebyshev": _chebyshev_distance,
}
