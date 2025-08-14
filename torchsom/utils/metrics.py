"""Utility functions for metrics."""

import warnings
from typing import Callable

import torch

from ..utils.grid import axial_distance, convert_to_axial_coords


def calculate_quantization_error(
    data: torch.Tensor,
    weights: torch.Tensor,
    distance_fn: Callable,
) -> float:
    """Calculate quantization error for a SOM.

    Args:
        data (torch.Tensor): Input data tensor [batch_size, num_features] or [num_features]
        weights (torch.Tensor): SOM weights [row_neurons, col_neurons, num_features]
        distance_fn (Callable): Function to compute distances between data and weights

    Returns:
        float: Average quantization error value
    """
    # Ensure batch compatibility
    device = weights.device
    data = data.to(device)
    if data.dim() == 1:
        data = data.unsqueeze(0)

    # Reshape for distance calculation
    data_expanded = data.view(data.shape[0], 1, 1, -1)
    weights_expanded = weights.unsqueeze(0)

    # Calculate distances between each data point and all neurons
    distances = distance_fn(data_expanded, weights_expanded)

    # Calculate minimum distance for each data point
    min_distances = torch.min(distances.view(distances.shape[0], -1), dim=1)[0]

    # Return average quantization error
    return min_distances.mean().item()


def calculate_topographic_error(
    data: torch.Tensor,
    weights: torch.Tensor,
    distance_fn: Callable,
    topology: str = "rectangular",
    # xx: torch.Tensor = None,
    # yy: torch.Tensor = None,
) -> float:
    """Calculate topographic error for a SOM.

    Args:
        data (torch.Tensor): Input data tensor [batch_size, num_features] or [num_features]
        weights (torch.Tensor): SOM weights [row_neurons, col_neurons, num_features]
        distance_fn (Callable): Function to compute distances between data and weights
        topology (str, optional): Grid configuration. Defaults to "rectangular".
        # xx (torch.Tensor, optional): Meshgrid of x coordinates. Required for hexagonal topology. Defaults to None.
        # yy (torch.Tensor, optional): Meshgrid of y coordinates. Required for hexagonal topology. Defaults to None.

    Returns:
        float: Topographic error ratio
    """
    # Ensure batch compatibility
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

    # Reshape for distance calculation
    data_expanded = data.view(data.shape[0], 1, 1, -1)
    weights_expanded = weights.unsqueeze(0)

    # Calculate distances between each data point and all neurons
    distances = distance_fn(data_expanded, weights_expanded)

    # ! Modification to test: all the lines below could be vectorized
    # Get top 2 BMU indices for each sample
    batch_size = distances.shape[0]
    _, indices = torch.topk(distances.view(batch_size, -1), k=2, largest=False, dim=1)

    # Compute topographic error based on topology
    if topology == "hexagonal":
        # Implement hexagonal topographic error
        error_count = 0
        for i in range(batch_size):
            # Convert flattened indices to 2D coordinates
            bmu1_row = int(torch.div(indices[i, 0], y_dim, rounding_mode="floor"))
            bmu1_col = int(indices[i, 0] % y_dim)
            bmu2_row = int(torch.div(indices[i, 1], y_dim, rounding_mode="floor"))
            bmu2_col = int(indices[i, 1] % y_dim)

            q1, r1 = convert_to_axial_coords(bmu1_row, bmu1_col)
            q2, r2 = convert_to_axial_coords(bmu2_row, bmu2_col)

            # Calculate distance in hex steps
            hex_distance = axial_distance(q1, r1, q2, r2)

            # Count as error if not neighbors (distance > 1)
            if hex_distance > 1:
                error_count += 1

        return error_count / batch_size
    else:
        # Implement rectangular topographic error
        threshold = 1.0  # Consider only direct neighbors (4-connectivity)

        # Convert flattened indices to 2D row, col coordinates
        bmu_row = torch.div(indices, y_dim, rounding_mode="floor")
        bmu_col = indices % y_dim

        # Calculate distances between best and second-best BMUs
        dx = bmu_row[:, 1] - bmu_row[:, 0]
        dy = bmu_col[:, 1] - bmu_col[:, 0]
        distances = torch.sqrt(dx.float() ** 2 + dy.float() ** 2)

        # Units are not neighbors if distance > threshold
        return (distances > threshold).float().mean().item()
