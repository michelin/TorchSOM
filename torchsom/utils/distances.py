from typing import Optional

import torch


def _cosine_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine distance between input and weights.

    Args:
        data (torch.Tensor): input data tensor of shape [batch_size, 1, 1, n_features]
        weights (torch.Tensor): SOM weights tensor of shape [1, row_neurons, col_neurons, n_features]

    Returns:
        torch.Tensor: cosine distance between input and weights [batch_size, row_neurons, col_neurons]
    """

    # Normalize vectors to unit length for numerical stability
    eps = 1e-8
    data_normalized = data / (torch.norm(data, dim=-1, keepdim=True) + eps)
    weights_normalized = weights / (torch.norm(weights, dim=-1, keepdim=True) + eps)

    # Compute cosine similarity
    cos_sim = torch.sum(data_normalized * weights_normalized, dim=-1)

    # Convert to distance (1 - similarity) and ensure values are in [0, 1]
    cos_dist = torch.clamp(1 - cos_sim, min=0.0, max=1.0)

    return cos_dist


def _euclidean_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute Euclidean distance between input and weights.

    Args:
        data (torch.Tensor): input data tensor of shape [batch_size, 1, 1, n_features]
        weights (torch.Tensor): SOM weights tensor of shape [1, row_neurons, col_neurons, n_features]

    Returns:
        torch.Tensor: euclidean distance between input and weights [batch_size, row_neurons, col_neurons]
    """
    return torch.norm(torch.subtract(data, weights), dim=-1)


def _manhattan_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute Manhattan distance between input and weights.

        Args:
            data (torch.Tensor): input data tensor of shape [batch_size, 1, 1, n_features]
            weights (torch.Tensor): SOM weights tensor of shape [1, row_neurons, col_neurons, n_features]

    Returns:
            torch.Tensor: manhattan distance between input and weights [batch_size, row_neurons, col_neurons]
    """

    return torch.norm(torch.subtract(data, weights), p=1, dim=-1)


def _chebyshev_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute Chebyshev distance between input and weights.

        Args:
            data (torch.Tensor): input data tensor of shape [batch_size, 1, 1, n_features]
            weights (torch.Tensor): SOM weights tensor of shape [1, row_neurons, col_neurons, n_features]

    Returns:
            torch.Tensor: chebyshev distance between input and weights [batch_size, row_neurons, col_neurons]
    """

    # return torch.max(torch.subtract(data, weights), dim=-1).values
    return torch.max(torch.abs(data - weights), dim=-1).values


def _weighted_euclidean_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
    weight_proportions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute weighted Euclidean distance between input and weights.

    Args:
        data (torch.Tensor): Input data tensor of shape [batch_size, 1, 1, n_features]
        weights (torch.Tensor): SOM weights tensor of shape [1, row_neurons, col_neurons, n_features]
        weight_proportions (Optional[torch.Tensor], optional): Feature weights tensor of shape [n_features] or broadcastable shape. Defaults to None.

    Returns:
        torch.Tensor: Weighted Euclidean distance between input and weights [batch_size, row_neurons, col_neurons]

    Raises:
        ValueError: If weight_proportions shape is incompatible with feature dimensions
        ValueError: If weight_proportions contains non-positive values
    """
    # Validate inputs
    if weight_proportions is not None:
        # Ensure weight_proportions is on the same device
        weight_proportions = weight_proportions.to(data.device)

        # Check for non-positive weights
        if torch.any(weight_proportions <= 0):
            raise ValueError("All weight proportions must be positive")

        # Validate shape compatibility
        expected_features = data.shape[-1]
        if weight_proportions.numel() != expected_features:
            # Try to reshape if possible
            try:
                weight_proportions = weight_proportions.view(expected_features)
            except RuntimeError:
                raise ValueError(
                    f"Weight proportions shape {weight_proportions.shape} incompatible with "
                    f"feature dimension {expected_features}"
                )

        # Ensure weight_proportions can be broadcast with the difference tensor
        # Shape should be [1, 1, 1, n_features] to match data and weights broadcasting
        weight_proportions = weight_proportions.view(1, 1, 1, -1)

    # Compute squared differences
    diff_squared = torch.pow(data - weights, 2)

    # Apply feature weights if provided
    if weight_proportions is not None:
        diff_squared = diff_squared * weight_proportions

    # Sum across feature dimension and take square root
    return torch.sqrt(torch.sum(diff_squared, dim=-1))


def _efficient_euclidean_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Calculate Euclidean distances using vectorized operations for better performance.

    This implementation uses the mathematical identity:
    ||x - w||² = ||x||² + ||w||² - 2⟨x, w⟩

    Args:
        data (torch.Tensor): Input data tensor of shape [batch_size, 1, 1, n_features]
        weights (torch.Tensor): SOM weights tensor of shape [1, row_neurons, col_neurons, n_features]

    Returns:
        torch.Tensor: Euclidean distance between input and weights [batch_size, row_neurons, col_neurons]
    """
    # Reshape for efficient computation
    batch_size = data.shape[0]
    n_features = data.shape[-1]
    n_neurons = weights.shape[1] * weights.shape[2]

    # Flatten spatial dimensions of weights and data
    data_flat = data.view(batch_size, n_features)  # [batch_size, n_features]
    weights_flat = weights.view(n_neurons, n_features)  # [n_neurons, n_features]

    # Compute squared norms
    data_norm_sq = torch.sum(data_flat**2, dim=1, keepdim=True)  # [batch_size, 1]
    weights_norm_sq = torch.sum(weights_flat**2, dim=1, keepdim=True)  # [n_neurons, 1]

    # Compute dot products
    dot_products = torch.mm(data_flat, weights_flat.T)  # [batch_size, n_neurons]

    # Apply distance formula: ||x - w||² = ||x||² + ||w||² - 2⟨x, w⟩
    distances_sq = data_norm_sq + weights_norm_sq.T - 2 * dot_products

    # Clamp to avoid numerical issues with negative values
    distances_sq = torch.clamp(distances_sq, min=0.0)

    # Take square root and reshape back to original spatial dimensions
    distances = torch.sqrt(distances_sq)
    return distances.view(batch_size, weights.shape[1], weights.shape[2])


DISTANCE_FUNCTIONS = {
    "euclidean": _euclidean_distance,
    "cosine": _cosine_distance,
    "manhattan": _manhattan_distance,
    "chebyshev": _chebyshev_distance,
    "weighted_euclidean": _weighted_euclidean_distance,
    "efficient_euclidean": _efficient_euclidean_distance,
}
