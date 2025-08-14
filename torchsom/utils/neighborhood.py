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

        c (Tuple[int, int]): center of winning neuron coordinates [row, col]
        sigma (float): width of the neighborhood, so standard deviation. It controls the spread of the update influence.

    Returns:
        torch.Tensor: Gaussian neighborhood weights. Element-wise product standing for the combined influence of gaussian neighborhood around center c with a spread sigma [row_neurons, col_neurons].
    """
    denum = 2 * sigma * sigma
    gaussian_x = torch.exp(
        -torch.pow(xx - c[0], 2) / denum
    )  # Gaussian function in x direction [row_neurons, col_neurons]
    gaussian_y = torch.exp(
        -torch.pow(yy - c[1], 2) / denum
    )  # Gaussian function in y direction [row_neurons, col_neurons]
    return gaussian_x * gaussian_y


def _mexican_hat(
    xx: torch.Tensor,
    yy: torch.Tensor,
    c: tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    """Mexican hat (Ricker wavelet) neighborhood function to update weights.

    See also: https://en.wikipedia.org/wiki/Ricker_wavelet

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

        c (Tuple[int, int]): center of winning neuron coordinates [row, col]
        sigma (float): width of the neighborhood, so standard deviation. It controls the spread of the update influence.

    Returns:
        torch.Tensor: Mexican hat neighborhood weights. Element-wise product standing for the combined influence of mexican neighborhood around center c with a spread sigma [row_neurons, col_neurons].
    """
    denum = 2 * sigma * sigma
    cst = 1 / (torch.pi * torch.pow(torch.tensor(sigma), 4))
    squared_distances = torch.pow(xx - c[0], 2) + torch.pow(
        yy - c[1], 2
    )  # Squared distances from center [row_neurons, col_neurons]
    exp_distances = torch.exp(-squared_distances / denum)
    mexican_hat = cst * (1 - (1 / 2) * squared_distances / (2 * denum)) * exp_distances

    # ! Modification to test
    # denum = 2 * sigma * sigma
    # sigma_t = torch.tensor(sigma, device=xx.device, dtype=xx.dtype)
    # cst = 1 / (torch.pi * sigma_t.pow(4))
    # squared_distances = torch.pow(xx - c[0], 2) + torch.pow(
    #     yy - c[1], 2
    # )  # Squared distances from center [row_neurons, col_neurons]
    # exp_distances = torch.exp(-squared_distances / denum)
    # mexican_hat = cst * (1 - (1 / 2) * squared_distances / (2 * denum)) * exp_distances

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

        c (Tuple[int, int]): center of winning neuron coordinates [row, col]
        sigma (float): width of the neighborhood, so standard deviation. It controls the spread of the update influence.

    Returns:
        torch.Tensor: Binary bubble neighborhood weights. Mask to update elements only striclty within the sigma radius, hence bubble name [row_neurons, col_neurons].
    """
    x_distance, y_distance = torch.abs(xx - c[0]), torch.abs(
        yy - c[1]
    )  # Both [row_neurons, col_neurons]

    mask = (x_distance <= sigma) & (
        y_distance <= sigma
    )  # Binary mask set to true if both distances are below sigma for each neuron [row_neurons, col_neurons]
    return mask.float()  # Convert binary (True/False) to float (1./0.)


def _triangle(
    xx: torch.Tensor,
    yy: torch.Tensor,
    c: tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    """Triangle (linear) neighborhood function to update weights.

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

        c (Tuple[int, int]): center of winning neuron coordinates [row, col]
        sigma (float): width of the neighborhood, so standard deviation. It controls the spread of the update influence.

    Returns:
        torch.Tensor: Triangle neighborhood weights. Element-wise product standing for the combined influence of gaussian neighborhood around center c with a spread sigma [row_neurons, col_neurons].
    """
    triangle_x, triangle_y = (-torch.abs(c[0] - xx)) + sigma, (
        -torch.abs(c[1] - yy)
    ) + sigma  # Linear decay in both directions, both [row_neurons, col_neurons]

    triangle_x, triangle_y = torch.clamp(triangle_x, min=0.0), torch.clamp(
        triangle_y, min=0.0
    )  # Clip negative values to zero, both [row_neurons, col_neurons]
    return triangle_x * triangle_y


NEIGHBORHOOD_FUNCTIONS = {
    "gaussian": _gaussian,
    "mexican_hat": _mexican_hat,
    "bubble": _bubble,
    "triangle": _triangle,
}
