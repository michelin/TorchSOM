"""Utility functions for neighborhood functions."""

import torch


def _gaussian(
    xx: torch.Tensor,
    yy: torch.Tensor,
    c: tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    """Gaussian neighborhood function to update weights.

    See also: https://en.wikipedia.org/wiki/Gaussian_function
    Uses proper Euclidean distance in coordinate space, works for both rectangular and hexagonal topologies.

    Args:
        xx (torch.Tensor): Meshgrid of x coordinates [row_neurons, col_neurons]
        tensor(
            [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],

            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],

            ... ,

            [row_neurons, row_neurons, row_neurons, row_neurons, row_neurons, row_neurons, row_neurons]]
        )

        yy (torch.Tensor): Meshgrid of y coordinates [row_neurons, col_neurons]
        tensor(
            [[ 0.,  1.,  2.,  ...,  col_neurons],

            [ 0.,  1.,  2.,  ...,  col_neurons],

            ... ,

            [ 0.,  1.,  2.,  ...,  col_neurons],]
        )

        c (Tuple[int, int]): center of winning neuron coordinates [row, col] (grid indices)
        sigma (float): width of the neighborhood, so standard deviation. It controls the spread of the update influence.

    Returns:
        torch.Tensor: Gaussian neighborhood weights based on Euclidean distance [row_neurons, col_neurons]. Element-wise product standing for the combined influence of gaussian neighborhood around center c with a spread sigma [row_neurons, col_neurons].
    """
    # Get the coordinate of the center neuron from the meshgrid
    center_x = xx[c[0], c[1]]
    center_y = yy[c[0], c[1]]

    # Calculate squared Euclidean distances from center
    dx = xx - center_x
    dy = yy - center_y
    squared_distances = dx * dx + dy * dy

    # Apply Gaussian function based on Euclidean distance
    return torch.exp(-squared_distances / (2 * sigma * sigma))


def _mexican_hat(
    xx: torch.Tensor,
    yy: torch.Tensor,
    c: tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    """Mexican hat (Ricker wavelet) neighborhood function to update weights.

    See also: https://en.wikipedia.org/wiki/Ricker_wavelet
    Uses proper Euclidean distance in coordinate space, works for both rectangular and hexagonal topologies.

    Args:
        xx (torch.Tensor): Meshgrid of x coordinates [row_neurons, col_neurons]
        tensor(
            [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],

            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],

            ... ,

            [row_neurons, row_neurons, row_neurons, row_neurons, row_neurons, row_neurons, row_neurons]]
        )

        yy (torch.Tensor): Meshgrid of y coordinates [row_neurons, col_neurons]
        tensor(
            [[ 0.,  1.,  2.,  ...,  col_neurons],

            [ 0.,  1.,  2.,  ...,  col_neurons],

            ... ,

            [ 0.,  1.,  2.,  ...,  col_neurons],]
        )

        c (Tuple[int, int]): center of winning neuron coordinates [row, col] (grid indices)
        sigma (float): width of the neighborhood, so standard deviation. It controls the spread of the update influence.

    Returns:
        torch.Tensor: Mexican hat neighborhood weights based on Euclidean distance [row_neurons, col_neurons]. Element-wise product standing for the combined influence of mexican neighborhood around center c with a spread sigma [row_neurons, col_neurons].
    """
    # Get the coordinate of the center neuron from the meshgrid
    center_x = xx[c[0], c[1]]
    center_y = yy[c[0], c[1]]

    # Calculate squared Euclidean distances from center
    dx = xx - center_x
    dy = yy - center_y
    squared_distances = dx * dx + dy * dy

    # Mexican hat parameters
    denum = 2 * sigma * sigma
    sigma_tensor = torch.tensor(sigma, device=xx.device, dtype=xx.dtype)
    cst = 1 / (torch.pi * sigma_tensor.pow(4))

    # Apply Mexican hat function
    exp_distances = torch.exp(-squared_distances / denum)
    mexican_hat = cst * (1 - (1 / 2) * squared_distances / (2 * denum)) * exp_distances

    # Ensure the central peak is exactly 1.0
    max_value = mexican_hat[c[0], c[1]]
    if max_value > 0:
        mexican_hat = mexican_hat / max_value
    return mexican_hat


def _bubble(
    xx: torch.Tensor,
    yy: torch.Tensor,
    c: tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    """Bubble (step function) neighborhood function to update weights.

    Uses proper Euclidean distance in coordinate space, works for both rectangular and hexagonal topologies.

    Args:
        xx (torch.Tensor): Meshgrid of x coordinates [row_neurons, col_neurons]
        tensor(
            [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],

            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],

            ... ,

            [row_neurons, row_neurons, row_neurons, row_neurons, row_neurons, row_neurons, row_neurons]]
        )

        yy (torch.Tensor): Meshgrid of y coordinates [row_neurons, col_neurons]
        tensor(
            [[ 0.,  1.,  2.,  ...,  col_neurons],

            [ 0.,  1.,  2.,  ...,  col_neurons],

            ... ,

            [ 0.,  1.,  2.,  ...,  col_neurons],]
        )

        c (Tuple[int, int]): center of winning neuron coordinates [row, col] (grid indices)
        sigma (float): width of the neighborhood, so standard deviation. It controls the spread of the update influence.

    Returns:
        torch.Tensor: Binary bubble neighborhood weights based on Euclidean distance [row_neurons, col_neurons]. Mask to update elements only striclty within the sigma radius, hence bubble name [row_neurons, col_neurons].
    """
    # Get the coordinate of the center neuron from the meshgrid
    center_x = xx[c[0], c[1]]
    center_y = yy[c[0], c[1]]

    # Calculate Euclidean distances from center
    dx = xx - center_x
    dy = yy - center_y
    distances = torch.sqrt(dx * dx + dy * dy)

    # Binary mask: 1.0 within sigma radius, 0.0 outside
    mask = distances <= sigma
    return mask.float()  # Convert binary (True/False) to float (1./0.)


def _triangle(
    xx: torch.Tensor,
    yy: torch.Tensor,
    c: tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    """Triangle (linear) neighborhood function to update weights.

    Uses proper Euclidean distance in coordinate space, works for both rectangular and hexagonal topologies.

    Args:
        xx (torch.Tensor): Meshgrid of x coordinates [row_neurons, col_neurons]
        tensor(
            [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],

            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],

            ... ,

            [row_neurons, row_neurons, row_neurons, row_neurons, row_neurons, row_neurons, row_neurons]]
        )

        yy (torch.Tensor): Meshgrid of y coordinates [row_neurons, col_neurons]
        tensor(
            [[ 0.,  1.,  2.,  ...,  col_neurons],

            [ 0.,  1.,  2.,  ...,  col_neurons],

            ... ,

            [ 0.,  1.,  2.,  ...,  col_neurons],]
        )

        c (Tuple[int, int]): center of winning neuron coordinates [row, col] (grid indices)
        sigma (float): width of the neighborhood, so standard deviation. It controls the spread of the update influence.

    Returns:
        torch.Tensor: Triangle neighborhood weights based on Euclidean distance [row_neurons, col_neurons]. Element-wise product standing for the combined influence of triangle neighborhood around center c with a spread sigma [row_neurons, col_neurons].
    """
    # Get the coordinate of the center neuron from the meshgrid
    center_x = xx[c[0], c[1]]
    center_y = yy[c[0], c[1]]

    # Calculate Euclidean distances from center
    dx = xx - center_x
    dy = yy - center_y
    distances = torch.sqrt(dx * dx + dy * dy)

    # Linear decay from 1.0 at center to 0.0 at distance sigma
    triangle_weights = torch.clamp(sigma - distances, min=0.0) / sigma
    return triangle_weights


NEIGHBORHOOD_FUNCTIONS = {
    "gaussian": _gaussian,
    "mexican_hat": _mexican_hat,
    "bubble": _bubble,
    "triangle": _triangle,
}
