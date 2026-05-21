"""Utility functions for initialization."""

import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def random_init_dataloader(
    weights: torch.Tensor,
    dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Initialize SOM weights via reservoir sampling (Vitter's Algorithm R) over a DataLoader.

    Selects n_neurons samples uniformly from the stream without loading all data into memory.

    Args:
        weights (torch.Tensor): Weight tensor to initialize [row_neurons, col_neurons, num_features]
        dataloader (DataLoader): DataLoader providing batches of data
        device (str, optional): Device for tensor computations. Defaults to "cuda" if available, else "cpu".

    Returns:
        torch.Tensor: Initialized weights
    """
    n_neurons = weights.shape[0] * weights.shape[1]
    n_features = weights.shape[2]

    reservoir = torch.empty(n_neurons, n_features, device=device)
    global_idx = 0  # total samples seen so far

    for batch in tqdm(dataloader, desc="Initializing weights", total=len(dataloader)):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device)

        if global_idx == 0 and batch.shape[1] != n_features:
            raise ValueError(
                f"DataLoader feature dim ({batch.shape[1]}) doesn't match weights ({n_features})"
            )

        batch_size = batch.shape[0]
        batch_offset = 0

        # Fill phase: directly populate reservoir until full
        if global_idx < n_neurons:
            fill_count = min(n_neurons - global_idx, batch_size)
            reservoir[global_idx : global_idx + fill_count] = batch[:fill_count]
            global_idx += fill_count
            batch_offset = fill_count

        # Replacement phase: vectorized reservoir sampling for remaining samples
        n_remaining = batch_size - batch_offset
        if n_remaining > 0:
            remaining = batch[batch_offset:]
            sample_indices = torch.arange(
                global_idx, global_idx + n_remaining, device=device
            )
            # For each sample at global index i, draw j in [0, i]
            rand_draws = (
                torch.rand(n_remaining, device=device) * (sample_indices + 1).float()
            ).long()
            accept_mask = rand_draws < n_neurons
            if accept_mask.any():
                reservoir[rand_draws[accept_mask]] = remaining[accept_mask]
            global_idx += n_remaining

    return reservoir.view(weights.shape[0], weights.shape[1], n_features)


def incremental_pca_init(
    weights: torch.Tensor,
    dataloader: DataLoader,
    topology: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Initialize SOM weights using PCA computed incrementally via Chan's parallel algorithm.

    Accumulates mean and covariance matrix in O(F²) memory without storing data.

    Args:
        weights (torch.Tensor): Weight tensor to initialize [row_neurons, col_neurons, num_features]
        dataloader (DataLoader): DataLoader providing batches of data
        topology (str): Grid configuration, "rectangular" or "hexagonal"
        device (str, optional): Device for tensor computations. Defaults to "cuda" if available, else "cpu".

    Returns:
        torch.Tensor: Initialized weights
    """
    n_features = weights.shape[2]

    if n_features == 1:
        raise ValueError("Data needs at least 2 features for PCA initialization")

    if weights.shape[0] == 1 or weights.shape[1] == 1:
        warnings.warn(
            "PCA initialization may be inappropriate for 1D map",
            stacklevel=2,
        )

    # Chan's parallel algorithm accumulators
    n_total = 0
    mean = torch.zeros(n_features, device=device)
    M2 = torch.zeros(n_features, n_features, device=device)

    for batch in tqdm(dataloader, desc="Initializing weights", total=len(dataloader)):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device)

        if n_total == 0 and batch.shape[1] != n_features:
            raise ValueError(
                f"DataLoader feature dim ({batch.shape[1]}) doesn't match weights ({n_features})"
            )

        n_b = batch.shape[0]
        mean_b = batch.mean(dim=0)
        centered_b = batch - mean_b.unsqueeze(0)
        M2_b = centered_b.T @ centered_b

        # Merge batch stats with running stats (Chan's formula)
        n_ab = n_total + n_b
        delta = mean_b - mean
        mean = mean + delta * (n_b / n_ab)
        M2 = M2 + M2_b + torch.outer(delta, delta) * (n_total * n_b / n_ab)
        n_total = n_ab

    if n_total < 2:
        raise ValueError("Cannot perform PCA: need at least 2 samples total")

    cov = M2 / (n_total - 1)

    U, S, V = torch.linalg.svd(cov, full_matrices=True)
    pc = V[:2]  # Top 2 principal components

    # Create coordinate grid for initialization
    x_coords = torch.linspace(-1, 1, weights.shape[0], device=device)
    y_coords = torch.linspace(-1, 1, weights.shape[1], device=device)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
    adj_grid_x, adj_grid_y = adjust_meshgrid_topology(
        xx=grid_x, yy=grid_y, topology=topology
    )

    pca_weights = adj_grid_x.unsqueeze(-1) * pc[0].unsqueeze(0).unsqueeze(
        0
    ) + adj_grid_y.unsqueeze(-1) * pc[1].unsqueeze(0).unsqueeze(0)

    # Scale using diagonal of covariance as std approximation
    data_std_approx = cov.diagonal().mean().sqrt()
    weights_std = pca_weights.std()
    if weights_std > 0:
        pca_weights = pca_weights * (data_std_approx / weights_std)

    return pca_weights + mean.unsqueeze(0).unsqueeze(0)


def initialize_weights(
    weights: torch.Tensor,
    data: torch.Tensor | DataLoader,
    mode: str = "random",
    topology: str = "rectangular",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Main function to initialize weights based on specified method.

    Args:
        weights (torch.Tensor): Weight tensor to initialize [row_neurons, col_neurons, num_features]
        data (torch.Tensor | DataLoader): Input data tensor [batch_size, num_features] or DataLoader
        mode (str, optional): Initialization method, "random" or "pca". Defaults to "random".
        topology (str, optional): Grid configuration, "rectangular" or "hexagonal". Defaults to "rectangular".
        device (str, optional): Device for tensor computations. Defaults to "cuda" if available, else "cpu".

    Returns:
        torch.Tensor: Initialized weights

    Raises:
        ValueError: If an invalid initialization mode is provided
    """
    if isinstance(data, DataLoader):
        if mode == "random":
            return random_init_dataloader(weights, data, device)
        elif mode == "pca":
            return incremental_pca_init(weights, data, topology, device)
        else:
            raise ValueError(
                "The only method to initialize the weights are 'random' or 'pca'."
            )

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
