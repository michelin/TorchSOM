"""Utility functions for neighborhood functions."""

import torch


def _gaussian(
    coord_distances_sq: torch.Tensor,
    bmu_indices: torch.Tensor,
    sigma: float,
    x: int,
    y: int,
) -> torch.Tensor:
    """Compute Gaussian neighborhood for batch of BMUs efficiently to update weights.

    See also: https://en.wikipedia.org/wiki/Gaussian_function

    Args:
        coord_distances_sq (torch.Tensor): Precomputed squared coordinate distances in the grid space [x*y, x*y]
        bmu_indices (torch.Tensor): Tensor of shape [batch_size] with flattened BMU indices
        sigma (float): current sigma value, width of the neighborhood, so standard deviation. It controls the spread of the update influence.
        x (int): grid width
        y (int): grid height

    Returns:
        Neighborhood weights of shape [batch_size, x, y]
    """
    batch_size = bmu_indices.shape[0]
    # Retrieve distance matrix for BMU positions: [batch_size, x*y]
    distances_sq = coord_distances_sq[bmu_indices]
    # Apply Gaussian function: [batch_size, x*y]
    neighborhood_flat = torch.exp(-distances_sq / (2 * sigma * sigma))
    # Reshape to grid: [batch_size, x, y]
    return neighborhood_flat.view(batch_size, x, y)


def _mexican_hat(
    coord_distances_sq: torch.Tensor,
    bmu_indices: torch.Tensor,
    sigma: float,
    x: int,
    y: int,
) -> torch.Tensor:
    """Compute Mexican hat (Ricker wavelet) neighborhood for batch of BMUs efficiently to update weights.

    See also: https://en.wikipedia.org/wiki/Ricker_wavelet

    Args:
        coord_distances_sq (torch.Tensor): Precomputed squared coordinate distances in the grid space [x*y, x*y]
        bmu_indices (torch.Tensor): Tensor of shape [batch_size] with flattened BMU indices
        sigma (float): current sigma value, width of the neighborhood, so standard deviation. It controls the spread of the update influence.
        x (int): grid width
        y (int): grid height

    Returns:
        Neighborhood weights of shape [batch_size, x, y]
    """
    batch_size = bmu_indices.shape[0]
    # Retrieve distance matrix for BMU positions: [batch_size, x*y]
    distances_sq = coord_distances_sq[bmu_indices]
    # Mexican hat parameters
    denum = 2 * sigma * sigma
    cst = 1.0 / (torch.pi * sigma**4)
    # Calculate Mexican hat function: [batch_size, x*y]
    exp_distances = torch.exp(-distances_sq / denum)
    mexican_hat = cst * (1 - 0.25 * distances_sq / (sigma * sigma)) * exp_distances
    # Normalize central peak to 1.0
    max_values = mexican_hat.max(dim=1, keepdim=True).values
    mexican_hat = torch.where(max_values > 0, mexican_hat / max_values, mexican_hat)
    # Reshape to grid: [batch_size, x, y]
    return mexican_hat.view(batch_size, x, y)


def _bubble(
    coord_distances_sq: torch.Tensor,
    bmu_indices: torch.Tensor,
    sigma: float,
    x: int,
    y: int,
) -> torch.Tensor:
    """Compute Bubble (step function) neighborhood for batch of BMUs efficiently to update weights.

    Args:
        coord_distances_sq (torch.Tensor): Precomputed squared coordinate distances in the grid space [x*y, x*y]
        bmu_indices (torch.Tensor): Tensor of shape [batch_size] with flattened BMU indices
        sigma (float): current sigma value, width of the neighborhood, so standard deviation. It controls the spread of the update influence.
        x (int): grid width
        y (int): grid height

    Returns:
        Neighborhood weights of shape [batch_size, x, y]
    """
    batch_size = bmu_indices.shape[0]
    # Retrieve distance matrix for BMU positions: [batch_size, x*y]
    distances_sq = coord_distances_sq[bmu_indices]
    # Create binary mask: [batch_size, x*y]
    mask = distances_sq <= sigma**2
    # Reshape to grid: [batch_size, x, y]
    return mask.float().view(batch_size, x, y)


def _triangle(
    coord_distances_sq: torch.Tensor,
    bmu_indices: torch.Tensor,
    sigma: float,
    x: int,
    y: int,
) -> torch.Tensor:
    """Compute Triangle (linear) neighborhood for batch of BMUs efficiently to update weights.

    Args:
        coord_distances_sq (torch.Tensor): Precomputed squared coordinate distances in the grid space [x*y, x*y]
        bmu_indices (torch.Tensor): Tensor of shape [batch_size] with flattened BMU indices
        sigma (float): current sigma value, width of the neighborhood, so standard deviation. It controls the spread of the update influence.
        x (int): grid width
        y (int): grid height

    Returns:
        Neighborhood weights of shape [batch_size, x, y]
    """
    batch_size = bmu_indices.shape[0]
    # Retrieve distance matrix for BMU positions: [batch_size, x*y]
    distances_sq = coord_distances_sq[bmu_indices]
    # Calculate triangle weights: [batch_size, x*y]
    distances = torch.sqrt(distances_sq)
    triangle_weights = torch.clamp(sigma - distances, min=0.0) / sigma
    # Reshape to grid: [batch_size, x, y]
    return triangle_weights.view(batch_size, x, y)


NEIGHBORHOOD_FUNCTIONS = {
    "gaussian": _gaussian,
    "mexican_hat": _mexican_hat,
    "bubble": _bubble,
    "triangle": _triangle,
}
