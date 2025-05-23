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
        torch.Tensor: euclidean distance between input and weights [row_neurons, col_neurons]
    """

    return torch.max(torch.abs(data - weights), dim=-1).values


def _manhattan_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute Manhattan distance between input and weights.

    Args:
        data (torch.Tensor): input data tensor of shape [batch_size, 1, 1, n_features]
        weights (torch.Tensor): SOM weights tensor of shape [1, row_neurons, col_neurons, n_features]

    Returns:
        torch.Tensor: manhattan distance between input and weights [row_neurons, col_neurons]
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
        torch.Tensor: chebyshev distance between input and weights [row_neurons, col_neurons]
    """

    # return torch.max(torch.subtract(data, weights), dim=-1).values
    return torch.max(torch.abs(data - weights), dim=-1).values


# TODO Check if this method works and if it is more efficient (also ensure it is compatible with batch framework)
def _efficient_euclidean_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # """Calculate Euclidean distances between input data and all neurons' weights.

    # Args:
    #     data (torch.Tensor): input data tensor [num_samples, num_features]

    # Returns:
    #     torch.Tensor: distance between input data and weights [num_samples, row_neurons * col_neurons]
    # """

    input_data_sq = torch.pow(data, 2).sum(
        dim=1, keepdim=True
    )  # Sum along columns [num_samples]
    weights_flat = weights.reshape(
        -1, weights.shape[2]
    )  # Convert [row_neurons, col_neurons, features] to [row_neurons * col_neurons, features]
    weights_flat_sq = torch.pow(weights_flat, 2).sum(
        dim=1, keepdim=True
    )  # Sum along columns [num_samples]
    dot_product = torch.mm(
        data, weights_flat.T
    )  # Dot product through matrix multiplication [num_samples, row_neurons * col_neurons]
    return torch.sqrt(
        -2 * dot_product + input_data_sq + weights_flat_sq.T
    )  # L2(x-w) = L2(x) + L2(w) - 2*dot_prod(x,w)


# TODO Check if this method works (also ensure it is compatible with batch framework)
def _weighted_euclidean_distance(
    data: torch.Tensor,
    weights: torch.Tensor,
    weights_proportion: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # """Compute weighted Euclidean distance between input and weights.

    # Args:
    #     data (torch.Tensor): input data [n_features]
    #     weights (torch.Tensor): SOM weights [row_neurons, col_neurons, n_features]
    #     weights_proportion (Optional[torch.Tensor], optional): Optional tensor to balance the weight of each feature [n_features]. Defaults to None.

    # Returns:
    #     torch.Tensor: weighted Euclidean distance between input and weights [row_neurons, col_neurons]
    # """

    distances_sq = torch.subtract(data, weights) ** 2
    if weights_proportion is not None:
        distances_sq = distances_sq * weights_proportion
    return torch.sqrt(torch.sum(distances_sq, dim=-1))


DISTANCE_FUNCTIONS = {
    "euclidean": _euclidean_distance,
    "cosine": _cosine_distance,
    "manhattan": _manhattan_distance,
    "chebyshev": _chebyshev_distance,
    "weighted_euclidean": _weighted_euclidean_distance,
    "efficient_euclidean": _efficient_euclidean_distance,
}
