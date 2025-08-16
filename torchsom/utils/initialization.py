"""Utility functions for initialization."""

import warnings

import torch

from torchsom.utils.grid import adjust_meshgrid_topology


def random_init(
    weights: torch.Tensor,
    data: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Initialize SOM weights by sampling random data points.

    Args:
        weights (torch.Tensor): Weight tensor to initialize [row_neurons, col_neurons, num_features]
        data (torch.Tensor): Input data tensor to sample from [batch_size, num_features]
        device (str, optional): Device for tensor computations. Defaults to "cuda" if available, else "cpu".

    Returns:
        torch.Tensor: Initialized weights
    """
    # Ensure data is on the correct device
    data = data.to(device)

    try:
        # Generate random indices for sampling
        indices = torch.randint(
            0, len(data), (weights.shape[0], weights.shape[1]), device=device
        )

        # Sample data points and assign to weights
        sampled_weights = data[indices]

        return sampled_weights

    except RuntimeError as e:
        raise RuntimeError(f"Random initialization failed: {str(e)}")


def pca_init(
    weights: torch.Tensor,
    data: torch.Tensor,
    topology: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Initialize SOM weights using PCA for faster convergence.

    Args:
        weights (torch.Tensor): Weight tensor to initialize [row_neurons, col_neurons, num_features]
        data (torch.Tensor): Input data tensor [batch_size, num_features]
        topology (str): Grid configuration, "rectangular" or "hexagonal"
        device (str, optional): Device for tensor computations. Defaults to "cuda" if available, else "cpu".

    Returns:
        torch.Tensor: Initialized weights
    """
    # Ensure data is on the correct device
    data = data.to(device)

    if weights.shape[2] == 1:
        raise ValueError("Data needs at least 2 features for PCA initialization")

    if weights.shape[0] == 1 or weights.shape[1] == 1:
        warnings.warn(
            "PCA initialization may be inappropriate for 1D map",
            stacklevel=2,
        )

    try:
        # Center the data efficiently using running mean
        data_mean = data.mean(dim=0, keepdim=True)
        data_centered = data - data_mean

        # Compute covariance matrix with improved numerical stability
        n_samples = len(data)
        if n_samples == 1:
            raise ValueError("Cannot perform PCA on a single sample")
        cov = torch.mm(data_centered.T, data_centered) / (n_samples - 1)

        # Try SVD first (more stable than eigendecomposition)
        try:
            U, S, V = torch.linalg.svd(
                cov,
                driver=None,  # Default is None, but also: "gesvd" (small), "gesvdj" (medium), and "gesvda" (large)
                full_matrices=True,  # Default is True
            )
            pc = V[:2]  # Take first two principal components

        except RuntimeError:
            warnings.warn(
                "SVD failed, falling back to eigendecomposition", stacklevel=2
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            idx = torch.argsort(
                eigenvalues, descending=True
            )  # Sort eigenvectors by eigenvalues in descending order
            pc = eigenvectors[
                :, idx[:2]
            ].T  # Works properly ! Results seems identical to driver=None

        # Create coordinate grid for initialization
        x_coords = torch.linspace(-1, 1, weights.shape[0], device=device)
        y_coords = torch.linspace(-1, 1, weights.shape[1], device=device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
        adj_grid_x, adj_grid_y = adjust_meshgrid_topology(
            xx=grid_x, yy=grid_y, topology=topology
        )

        # Initialize weights using broadcasting
        pca_weights = adj_grid_x.unsqueeze(-1) * pc[0].unsqueeze(0).unsqueeze(
            0
        ) + adj_grid_y.unsqueeze(-1) * pc[1].unsqueeze(0).unsqueeze(0)

        # Scale weights to match data distribution
        weights_std = pca_weights.std()
        if weights_std > 0:
            pca_weights = pca_weights * (data.std() / weights_std)

        # Add back the mean
        return pca_weights + data_mean

    except Exception as e:
        warnings.warn(
            f"PCA initialization failed: {str(e)}. Falling back to random initialization",
            stacklevel=2,
        )
        return random_init(weights, data, device)


def initialize_weights(
    weights: torch.Tensor,
    data: torch.Tensor,
    mode: str = "random",
    topology: str = "rectangular",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Main function to initialize weights based on specified method.

    Args:
        weights (torch.Tensor): Weight tensor to initialize [row_neurons, col_neurons, num_features]
        data (torch.Tensor): Input data tensor [batch_size, num_features]
        mode (str, optional): Initialization method, "random" or "pca". Defaults to "random".
        topology (str, optional): Grid configuration, "rectangular" or "hexagonal". Defaults to "rectangular".
        device (str, optional): Device for tensor computations. Defaults to "cuda" if available, else "cpu".

    Returns:
        torch.Tensor: Initialized weights

    Raises:
        ValueError: If an invalid initialization mode is provided
    """
    if data.shape[1] != weights.shape[2]:
        raise ValueError(
            f"Input data dimension ({data.shape[1]}) and weights dimension ({weights.shape[2]}) don't match"
        )

    if mode == "random":
        return random_init(weights, data, device)
    elif mode == "pca":
        return pca_init(weights, data, topology, device)
    else:
        raise ValueError(
            "The only method to initialize the weights are 'random' or 'pca'."
        )
