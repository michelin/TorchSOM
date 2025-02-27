import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .decay import DECAY_FUNCTIONS
from .distances import DISTANCE_FUNCTIONS
from .neighborhood import NEIGHBORHOOD_FUNCTIONS
from .utils import (
    adjust_meshgrid_topology,
    axial_distance,
    convert_to_axial_coords,
    create_mesh_grid,
)


class TorchSOM(nn.Module):
    """PyTorch implementation of Self Organizing Maps.

    Args:
        nn (_type_): PyTorch neural network module
    """

    def __init__(
        self,
        x: int,
        y: int,
        num_features: int,
        epochs: int = 10,
        batch_size: int = 5,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        topology: str = "rectangular",
        lr_decay_function: str = "asymptotic_decay",
        sigma_decay_function: str = "asymptotic_decay",
        neighborhood_function: str = "gaussian",
        distance_function: str = "euclidean",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        random_seed: int = 42,
    ):
        """Initialize the SOM.

        Args:
            x (int): Number of rows
            y (int): Number of cols
            num_features (int): Number of input features
            epochs (int, optional): Number of epochs to train. Defaults to 10.
            batch_size (int, optional): Number of samples to be considered at each epoch (training). Defaults to 5.
            sigma (float, optional): width of the neighborhood, so standard deviation. It controls the spread of the update influence. Defaults to 1.0.
            learning_rate (float, optional): strength of the weights updates. Defaults to 0.5.
            topology (str, optional): Grid configuration. Defaults to "rectangular".
            lr_decay_function (str, optional): Function to adjust (decrease) the learning rate at each epoch (training). Defaults to "asymptotic_decay".
            sigma_decay_function (str, optional): Function to adjust (decrease) the sigma at each epoch (training). Defaults to "asymptotic_decay".
            neighborhood_function (str, optional): Function to update the weights at each epoch (training). Defaults to "gaussian".
            distance_function (str, optional): Function to compute the distance between grid weights and input data. Defaults to "euclidean".
            device (str, optional): Allocate tensors on CPU or GPU. Defaults to "cuda" if available, else "cpu".
            random_seed (int, optional): Ensure reproducibility. Defaults to 42.

        Raises:
            ValueError: Ensure valid topology
        """
        super(TorchSOM, self).__init__()

        # Validate parameters
        if sigma > torch.sqrt(torch.tensor(float(x * x + y * y))):
            warnings.warn(
                "Warning: sigma might be too high for the dimension of the map."
            )
        if topology not in ["hexagonal", "rectangular"]:
            raise ValueError("Only hexagonal and rectangular topologies are supported")

        # Input parameters
        self.x = x
        self.y = y
        self.num_features = num_features
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.topology = topology
        self.random_seed = random_seed
        self.distance_fn_name = distance_function
        self.distance_fn = DISTANCE_FUNCTIONS[distance_function]
        self.lr_decay_fn = DECAY_FUNCTIONS[lr_decay_function]
        self.sigma_decay_fn = DECAY_FUNCTIONS[sigma_decay_function]

        # Set up x and y mesh grids, adjust them based on the topology
        x_meshgrid, y_meshgrid = create_mesh_grid(x, y, device)
        self.xx, self.yy = adjust_meshgrid_topology(x_meshgrid, y_meshgrid, topology)

        # Set up neighborhood function
        self.neighborhood_fn = lambda win_neuron, sigma: NEIGHBORHOOD_FUNCTIONS[
            neighborhood_function
        ](self.xx, self.yy, win_neuron, sigma)

        # Ensure reproducibility
        torch.manual_seed(random_seed)

        # Initialize & normalize weights
        weights = 2 * torch.randn(x, y, num_features, device=device) - 1
        normalized_weights = weights / torch.norm(weights, dim=-1, keepdim=True)
        self.weights = nn.Parameter(normalized_weights, requires_grad=False)

    """
    Helper methods
    """

    def _update_weights(
        self,
        data: torch.Tensor,
        bmus: Union[Tuple[int, int], torch.Tensor],
        learning_rate: float,
        sigma: float,
    ) -> None:
        """Update weights using neighborhood function. Handles both single samples and batches.

        Args:
            data (torch.Tensor): Input tensor of shape [num_features] or [batch_size, num_features]
            bmus (Union[Tuple[int, int], torch.Tensor]): BMU coordinates as tuple (single) or tensor (batch)
            learning_rate (float): Current learning rate
            sigma (float): Current sigma value
        """

        # Single sample
        if isinstance(bmus, tuple):

            # Calculate neighborhood contributions for the BMU and reshape for broadcasting
            neighborhood = self.neighborhood_fn(bmus, sigma)
            neighborhood = neighborhood.view(self.x, self.y, 1)

            # Calculate the update for the single sample
            update = learning_rate * neighborhood * (data - self.weights)

            # Update the weights
            self.weights.data += update

        # Batch samples
        else:

            # Calculate neighborhood contributions for each BMU in batch
            batch_size = data.shape[0]
            neighborhoods = torch.stack(
                [
                    self.neighborhood_fn((row.item(), col.item()), sigma)
                    for row, col in bmus
                ]
            )  # [batch_size, row_neurons, col_neurons]

            # Reshape for broadcasting
            neighborhoods = neighborhoods.view(batch_size, self.x, self.y, 1)
            data_expanded = data.view(batch_size, 1, 1, self.num_features)

            # Calculate the updates for all samples
            updates = learning_rate * neighborhoods * (data_expanded - self.weights)

            # Average updates across batch and apply to weights
            self.weights.data += updates.mean(dim=0)

    def _calculate_distances_to_neurons(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate distances between input data and all neurons' weights. Handles both single samples and batches.

        Args:
            data: Input tensor of shape [num_features] if single or [batch_size, num_features] if batch

        Returns:
            Distances tensor of shape [row_neurons, col_neurons] or [batch_size, row_neurons, col_neurons]
        """

        # Ensure device and batch compatibility
        # data = data.to(self.device)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        data_batch_size = data.shape[0]

        # Reshape both data and weights for broadcasting when calculating the distance
        data_expanded = data.view(
            data_batch_size, 1, 1, self.num_features
        )  # From [batch_size, num_features] to [batch_size, 1, 1, num_features]
        weights_expanded = self.weights.unsqueeze(
            0
        )  # [1, row_neurons, col_neurons, num_features]

        # Compute distances for the whole batch [batch_size, row_neurons, col_neurons]
        distances = self.distance_fn(data_expanded, weights_expanded)

        """
        With a single data point the distance need to be squeezed because it returns [batch_size, 1]. 
        However, there is only one sample, so let's just retrieve the scalar by removing the useless batch dimension.
        """
        if data_batch_size == 1:
            distances = distances.squeeze(0)

        return distances

    def _topographic_error_hexagonal(
        self,
        indices: torch.Tensor,
    ) -> float:
        """Calculate topographic error for hexagonal topology using axial coordinates.

        Args:
            indices (torch.Tensor): Tensor containing indices of best and second-best matching units [batch_size, 2]

        Returns:
            float: Topographic error ratio (hexagonal case)
        """

        batch_size = indices.shape[0]
        error_count = 0

        # Iterate over each sample in the batch
        for i in range(batch_size):

            # Get grid coordinates of best and second best
            bmu1_row = int(torch.div(indices[i, 0], self.y, rounding_mode="floor"))
            bmu1_col = int(indices[i, 0] % self.y)
            bmu2_row = int(torch.div(indices[i, 1], self.y, rounding_mode="floor"))
            bmu2_col = int(indices[i, 1] % self.y)

            # Convert to axial coordinates
            q1, r1 = convert_to_axial_coords(bmu1_row, bmu1_col)
            q2, r2 = convert_to_axial_coords(bmu2_row, bmu2_col)

            # Calculate distance in hex steps
            hex_distance = axial_distance(q1, r1, q2, r2)

            # Count as error if not neighbors (distance > 1)
            if hex_distance > 1:
                error_count += 1

        return error_count / batch_size

    def _topographic_error_rectangular(
        self,
        indices: torch.Tensor,
    ) -> float:
        """Calculate topographic error for rectangular topology.

        Args:
            indices (torch.Tensor): Tensor containing indices of best and second-best matching units [batch_size, 2]

        Returns:
            float: Topographic error ratio (rectangular case)
        """

        # sqrt(2) for diagonal neighbors in case of 8 neighbors, otherwise 1.0 if consider 4 neighbors (left, right, up, down)
        threshold = 1.0  # 1.42

        # Convert to x,y coordinates directly from the indices
        b2mu_x = torch.div(indices, self.y, rounding_mode="floor")
        b2mu_y = indices % self.y

        # Calculate distances between best and second-best
        dx = b2mu_x[:, 1] - b2mu_x[:, 0]
        dy = b2mu_y[:, 1] - b2mu_y[:, 0]
        distances = torch.sqrt(dx.float() ** 2 + dy.float() ** 2)

        # Units are not neighbors if distance > threshold
        return (distances > threshold).float().mean().item()

    """
    Public methods
    """

    def identify_bmus(
        self,
        data: torch.Tensor,
    ) -> Union[Tuple[int, int], torch.Tensor]:
        """Find BMUs for input data.  Handles both single samples and batches.

        Args:
            data (torch.Tensor): Input tensor of shape [num_features] or [batch_size, num_features]

        Returns:
            Union[Tuple[int, int], torch.Tensor]: For single sample: Tuple of (row, col). For batch: Tensor of shape [batch_size, 2] containing (row, col) pairs
        """

        distances = self._calculate_distances_to_neurons(data)

        # Unique sample [row_neurons, col_neurons]
        if distances.dim() == 2:
            index = torch.argmin(
                distances.view(-1)
            )  # From 2D tensor [m,n] to 1D tensor [m*n] then retrieve the index of the bmu with the smallest distance
            return tuple(
                torch.unravel_index(index, (self.x, self.y))
            )  # Convert the index to 2D coordinates

        # Batch samples [batch_size, row_neurons, col_neurons]
        else:
            indices = torch.argmin(
                distances.view(distances.shape[0], -1), dim=1
            )  # From 3D tensor [batch_size, m, n] to 2D tensor [batch_size, m*n] then retrieve the index of the bmu with the smallest distance for all samples
            return torch.stack(
                [torch.div(indices, self.y, rounding_mode="floor"), indices % self.y],
                dim=1,
            )

    def quantization_error(
        self,
        data: torch.Tensor,
    ) -> float:
        """Calculate quantization error.

        Args:
            data (torch.Tensor): input data tensor [batch_size, num_features] or [num_features]

        Returns:
            float: Average quantization error value
        """

        # Ensure device and batch compatibility
        data = data.to(self.device)
        if data.dim() == 1:
            data = data.unsqueeze(0)

        # Compute the distances between data samples and neurons [batch_size, row_neurons, col_neurons]
        distances = self._calculate_distances_to_neurons(data)

        return (
            torch.min(distances.view(distances.shape[0], -1), dim=1)[0].mean().item()
        )  # Average the minimum distances

    def topographic_error(
        self,
        data: torch.Tensor,
    ) -> float:
        """Calculate topographic error with batch support

        Args:
            data (torch.Tensor): input data tensor [batch_size, num_features] or [num_features]

        Returns:
            float: Topographic error ratio
        """

        # Ensure device and batch compatibility
        data = data.to(self.device)
        if data.dim() == 1:
            data = data.unsqueeze(0)

        if self.x * self.y == 1:
            warnings.warn("The topographic error is not defined for a 1-by-1 map.")
            return float("nan")

        # Compute the distances between data samples and neurons [batch_size, row_neurons, col_neurons]
        distances = self._calculate_distances_to_neurons(data)

        # Get top 2 BMU indices for each sample
        batch_size = distances.shape[0]
        _, indices = torch.topk(
            distances.view(batch_size, -1), k=2, largest=False, dim=1
        )

        # Compute topographic error based on topology
        if self.topology == "hexagonal":
            return self._topographic_error_hexagonal(indices)
        else:
            return self._topographic_error_rectangular(indices)

    def init_weights(
        self,
        data: torch.Tensor,
        mode: str = "random",
    ) -> None:
        """Data should be normalized before initialization. Initialize weights using

            1. Random samples from input data.
            2. PCA components to make the training process converge faster.

        Args:
            data (torch.Tensor): input data tensor [batch_size, num_features]
            mode (str, optional): selection of the method to init the weights. Defaults to "random".

        Raises:
            ValueError: Ensure neurons' weights and input data have the same number of features
            RuntimeError: If random initialization takes too long
            ValueError: Requires at least 2 features for PCA
            ValueError: Requires more than one sample to perform PCA
            ValueError: Ensure an appropriate method for initialization
        """

        if data.shape[1] != self.num_features:
            raise ValueError(
                f"Input data dimension ({data.shape[1]}) and weights dimension ({self.num_features}) don't match"
            )

        if mode == "random":
            try:
                # Generate random indices for sampling
                indices = torch.randint(
                    0, len(data), (self.x, self.y), device=self.device
                )

                # Sample data points and assign to weights and normalize
                self.weights.data = data[indices]
                # self.weights.data = self.weights.data / torch.norm(
                #     self.weights.data, dim=-1, keepdim=True
                # )

            except RuntimeError as e:
                raise RuntimeError(f"Random initialization failed: {str(e)}")

        elif mode == "pca":
            if self.num_features == 1:
                raise ValueError(
                    "Data needs at least 2 features for PCA initialization"
                )
            if self.x == 1 or self.y == 1:
                warnings.warn("PCA initialization may be inappropriate for 1D map")

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
                    warnings.warn("SVD failed, falling back to eigendecomposition")
                    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                    idx = torch.argsort(
                        eigenvalues, descending=True
                    )  # Sort eigenvectors by eigenvalues in descending order
                    pc = eigenvectors[
                        :, idx[:2]
                    ].T  # Works properly ! Results seems identical to driver=None

                # Create coordinate grid for initialization
                x_coords = torch.linspace(-1, 1, self.x, device=self.device)
                y_coords = torch.linspace(-1, 1, self.y, device=self.device)
                grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
                adj_grid_x, adj_grid_y = adjust_meshgrid_topology(
                    xx=grid_x, yy=grid_y, topology=self.topology
                )

                # Initialize weights using broadcasting for better efficiency => (x, y, 1) * (1, 1, features) -> (x, y, features)
                weights = adj_grid_x.unsqueeze(-1) * pc[0].unsqueeze(0).unsqueeze(
                    0
                ) + adj_grid_y.unsqueeze(-1) * pc[1].unsqueeze(0).unsqueeze(0)

                # Scale weights to match data distribution
                weights_std = weights.std()
                if weights_std > 0:
                    weights = weights * (data.std() / weights_std)

                # Add back the mean and normalize
                self.weights.data = weights + data_mean
                # self.weights.data = self.weights.data / torch.norm(
                #     self.weights.data, dim=-1, keepdim=True
                # )

            except Exception as e:
                warnings.warn(
                    f"PCA initialization failed: {str(e)}. Falling back to random initialization"
                )
                self.init_weights(data, mode="random")

        else:
            raise ValueError(
                "The only method to initialize the weights are 'random' or 'pca'."
            )

    def train_som(
        self,
        data: torch.Tensor,
    ) -> Tuple[List[float], List[float]]:
        """Train the SOM using batches and track errors.

        Args:
            data_tensor (torch.Tensor): input data tensor [batch_size, num_features]

        Returns:
            Tuple[List[float], List[float]]: Quantization and topographic errors [epoch]
        """

        data = data.to(self.device)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        q_errors = []
        t_errors = []

        for epoch in tqdm(
            range(self.epochs),
            desc="Training SOM",
            unit="epoch",
            disable=False,
        ):

            # Update learning parameters through decay function (schedulers)
            lr = self.lr_decay_fn(self.learning_rate, t=epoch, max_iter=self.epochs)
            sigma = self.sigma_decay_fn(self.sigma, t=epoch, max_iter=self.epochs)

            epoch_q_errors = []
            epoch_t_errors = []

            for batch in dataloader:
                batch_data = batch[0]

                # Get BMUs for all data points at once [batch_size, 2]
                bmus = self.identify_bmus(batch_data)

                # Update the weights of each neuron
                self._update_weights(batch_data, bmus, lr, sigma)

                # Calculate both errors at each batch and store them
                epoch_q_errors.append(self.quantization_error(batch_data))
                epoch_t_errors.append(self.topographic_error(batch_data))

            # Compute both average errors at each epoch and store them
            q_errors.append(torch.tensor(epoch_q_errors).mean().item())
            t_errors.append(100 * torch.tensor(epoch_t_errors).mean().item())

        return q_errors, t_errors

    def build_hit_map(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Returns a matrix where element i,j is the number of times that neuron i,j has been the winner.

        Args:
        data (torch.Tensor): input data tensor [batch_size, num_features]

        Returns:
        torch.Tensor: Matrix indicating the number of times each neuron has been identified as bmu. [row_neurons, col_neurons]
        """

        # Ensure device and batch compatibility
        data = data.to(self.device)
        if data.dim() == 1:
            data = data.unsqueeze(0)

        # Get BMUs for all data points at once [batch_size, 2]
        bmus = self.identify_bmus(data)

        # Build the map by counting the occurence of each bmu
        hit_map = torch.zeros((self.x, self.y), device=self.device)
        for row, col in bmus:
            hit_map[row, col] += 1

        return hit_map

    def build_bmus_map(
        self,
        data: torch.Tensor,
        return_indices: bool = False,
    ) -> Dict[Tuple[int, int], Any]:
        """Create a mapping of winning neurons to their corresponding data points.

        Args:
            data (torch.Tensor): input data tensor [batch_size, num_features] or [num_features]
            return_indices (bool, optional): If True, return indices instead of data points. Defaults to False.

        Returns:
            Dict[Tuple[int, int], Any]: Dictionary mapping bmus to data samples or indices
        """

        # Ensure device and batch compatibility
        data = data.to(self.device)
        if data.dim() == 1:
            data = data.unsqueeze(0)

        # Get BMUs for all data points at once [batch_size, 2]
        bmus = self.identify_bmus(data)

        # Build the map
        bmus_map = defaultdict(list)
        for idx, (row, col) in enumerate(bmus):

            # Convert BMU coordinates to integer tuple for dictionary key
            bmu_pos = (int(row.item()), int(col.item()))

            # Add the corresponding element to the dict
            if return_indices:
                bmus_map[bmu_pos].append(idx)
            else:
                bmus_map[bmu_pos].append(data[idx])

        # If you return data samples, then for each bmu you need to stack them to have a tensor of all data samples instead of multiple tensors
        if not return_indices:
            for bmu in bmus_map:
                bmus_map[bmu] = torch.stack(bmus_map[bmu])

        return bmus_map

    def build_rank_map(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Build a map of neuron ranks based on their target value standard deviations.

        Args:
            data (torch.Tensor): Input data tensor [batch_size, num_features]
            target (torch.Tensor): Labels tensor for data points [batch_size]

        Returns:
            torch.Tensor: Rank map where each neuron's value is its rank (1 = lowest std = best)
        """

        # Ensure device compatibility
        data = data.to(self.device)
        target = target.to(self.device)

        bmus_map = self.build_bmus_map(data, return_indices=True)
        neuron_stds = torch.full((self.x, self.y), float("nan"), device=self.device)

        # Calculate standard deviation for each neuron
        active_neurons = 0
        for bmu_pos, sample_indices in bmus_map.items():
            if len(sample_indices) > 0:
                active_neurons += 1

                # Consider neuron with multiple elements
                if len(sample_indices) > 1:
                    neuron_stds[bmu_pos] = torch.std(
                        target[sample_indices], unbiased=True
                    ).item()  # Use unbiased estimator for better small sample handling

                # Consider neuron with a unique element
                else:
                    neuron_stds[bmu_pos] = 0.0

        rank_map = torch.full((self.x, self.y), float("nan"), device=self.device)

        # Get mask to retrieve indices of non-NaN values
        valid_mask = ~torch.isnan(neuron_stds)
        valid_stds = neuron_stds[valid_mask]

        if len(valid_stds) > 0:
            # Sort stds in descending order and get ranks (+ 1 to make ranks 1-based)
            ranks = torch.argsort(valid_stds, descending=True).argsort() + 1

            # Ensure there are as many ranks as activated neurons
            assert (
                len(ranks) == active_neurons
            ), f"Rank count ({len(ranks)}) doesn't match active neurons ({active_neurons})"

            # Place ranks back in the map
            rank_map[valid_mask] = ranks.float()

        return rank_map

    def build_metric_map(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        reduction_parameter: str,
    ) -> torch.Tensor:
        """Calculate neurons' metrics based on target values.

        Args:
            data (torch.Tensor): Input data tensor [batch_size, num_features]
            target (torch.Tensor): Labels tensor for data points [batch_size]
            reduction_parameter (str): Decide the calculation to apply to each neuron, 'mean' or 'std'.

        Returns:
            torch.Tensor: Metric map based on the reduction parameter.
        """

        # Ensure device compatibility
        data = data.to(self.device)
        target = target.to(self.device)
        epsilon = 1e-8

        bmus_map = self.build_bmus_map(data, return_indices=True)
        metric_map = torch.full((self.x, self.y), float("nan"), device=self.device)

        # For each activated neurons, calculate the corresponding target metric
        for bmu_pos, samples_indices in bmus_map.items():
            if len(samples_indices) > 0:
                if reduction_parameter == "mean":
                    metric_map[bmu_pos] = torch.mean(target[samples_indices])
                elif reduction_parameter == "std":
                    # metric_map[bmu_pos] = torch.std(target[samples_indices])
                    if len(samples_indices) > 1:
                        metric_map[bmu_pos] = torch.std(
                            target[samples_indices], unbiased=True
                        )
                    else:
                        metric_map[bmu_pos] = (
                            epsilon  # To ensure visualization with a non-zero value
                        )
        return metric_map

    def build_score_map(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate neurons' score based on target values.

        Args:
            data (torch.Tensor): Input data tensor [batch_size, num_features]
            target (torch.Tensor): Labels tensor for data points [batch_size]

        Returns:
            torch.Tensor: Score map based on a chosen score function: std_neuron / sqrt(n_neuron) * log(N_data/n_neuron).
            The score combines the standard error with a term penalizing uneven sample distribution across neurons. Lower scores indicate better neuron representativeness.
        """

        # Ensure device compatibility
        data = data.to(self.device)
        target = target.to(self.device)
        epsilon = 1e-8

        bmus_map = self.build_bmus_map(data, return_indices=True)
        score_map = torch.full((self.x, self.y), float("nan"), device=self.device)

        # For each activated neurons, calculate the corresponding target metric
        for bmu_pos, samples_indices in bmus_map.items():
            if len(samples_indices) > 0:

                # Consider neuron with multiple elements
                if len(samples_indices) > 1:
                    std = torch.std(target[samples_indices], unbiased=True)
                    n_samples = torch.tensor(
                        len(samples_indices), dtype=torch.float32, device=self.device
                    )
                    total_samples = torch.tensor(
                        len(data), dtype=torch.float32, device=self.device
                    )
                    neuron_score = (std / torch.sqrt(n_samples)) * torch.log(
                        total_samples / n_samples
                    )

                # Consider neuron with a unique element
                else:
                    neuron_score = torch.tensor(
                        epsilon, dtype=torch.float32, device=self.device
                    )  # tensor to initialize tensor from scalars and ensure visualization with a non-zero value

                score_map[bmu_pos] = (
                    round(neuron_score.item(), 2) if neuron_score > epsilon else epsilon
                )
        return score_map

    def build_distance_map(
        self,
        scaling: str = "sum",
        distance_metric: str = None,
    ) -> torch.Tensor:
        """Computes the distance map of each neuron with its neighbors.

        The distance map represents the normalized sum or mean of distances
        between a neuron's weight vector and its neighboring neurons.

        Args:
            scaling (str, optional): Defaults to "sum".
                If 'mean', each cell is normalized by the average neighbor distance.
                If 'sum', normalization is done by the sum of distances.
            distance_metric (str, optional): _description_. Defaults to None.

        Raises:
            ValueError: If an invalid scaling option is provided.
            ValueError: If an invalid distance metric is provided.

        Returns:
            torch.Tensor: Normalized distance map [row_neurons, col_neurons]
        """

        if scaling not in ["sum", "mean"]:
            raise ValueError(
                f'scaling should be either "sum" or "mean" ({scaling} is not valid)'
            )

        # Indicate the distance function to use
        if distance_metric is None:
            distance_fn = self.distance_fn
        else:
            if distance_metric not in DISTANCE_FUNCTIONS:
                raise ValueError(f"Unsupported distance metric: {distance_metric}")
            distance_fn = DISTANCE_FUNCTIONS[distance_metric]

        # Initialize distance map based on topology
        num_neighbors = 6 if self.topology == "hexagonal" else 4
        distance_matrix = torch.full(
            (self.weights.shape[0], self.weights.shape[1], num_neighbors),
            float("nan"),
            device=self.device,
        )

        # Define neighbor patterns based on topology
        if self.topology == "hexagonal":
            # 6-connected neighbors for hexagonal topology while ensuring the offset patterns depending on row parity.
            even_row_offsets = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
            odd_row_offsets = [(0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)]
        else:
            # 4-connected neighbors for rectangular topology (von Neumann neighborhood)
            neighbor_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Compute distances for each neuron
        for row in range(self.weights.shape[0]):
            for col in range(self.weights.shape[1]):
                current_neuron = self.weights[row, col]

                # Select appropriate neighbor pattern
                if self.topology == "hexagonal":
                    offsets = even_row_offsets if row % 2 == 0 else odd_row_offsets
                else:
                    offsets = neighbor_offsets

                # Compute distances between curren neuron and its neighbors
                for neighbor_idx, (row_offset, col_offset) in enumerate(offsets):
                    neighbor_row = row + row_offset
                    neighbor_col = col + col_offset

                    # Ensure neighbor is within bounds to compute the distance
                    if (
                        0 <= neighbor_row < self.weights.shape[0]
                        and 0 <= neighbor_col < self.weights.shape[1]
                    ):
                        neighbor_neuron = self.weights[neighbor_row, neighbor_col]

                        """
                        Reshape weights to ensure batch compatibility with distance function => shape [a,b] becomes [1,a,b] after unsqueeze(0)
                        Each neuron has a shape of [num_features] so they become [1,num_features] and then [1,1,num_features]
                        Finally, distance function need to be squeezed because it returns [batch_size, 1] but there is only one sample, so let's just retrieve the scalar
                        """
                        solo_batch_current_neuron = current_neuron.unsqueeze(
                            0
                        ).unsqueeze(0)
                        solo_batch_neighbor_neuron = neighbor_neuron.unsqueeze(
                            0
                        ).unsqueeze(0)

                        # Calculate and store the distance between both neurons
                        distance_matrix[row, col, neighbor_idx] = distance_fn(
                            solo_batch_current_neuron,
                            solo_batch_neighbor_neuron,
                        ).squeeze()

        """
        Aggregate distances (either sum or mean). Each neuron has approximately k distances based on the topology (and bounds).
        Compute the aggregation on the last dimension where all the ,neighbor distances are computed.
        Both torch methods ignore NaNs.
        """
        if scaling == "mean":
            distance_matrix = torch.nanmean(distance_matrix, dim=2)
        else:
            distance_matrix = torch.nansum(distance_matrix, dim=2)

        # Normalize the distance map
        max_distance = torch.max(
            distance_matrix.masked_fill(torch.isnan(distance_matrix), float("-inf"))
        )  # Replace NaNs with -inf to be ignored by max()
        return distance_matrix / max_distance if max_distance > 0 else distance_matrix

    # TODO Re-implement properly this method: can be useful for classification tasks. Labels could be str or int.
    def build_labels_dict(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[Tuple[int, int], Any]:
        """Create a mapping of winning neurons to their corresponding label counts."""
        if len(data) != len(labels):
            raise ValueError("data and labels must have the same length.")

        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.identify_single_bmu(x)].append(l.item())

        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap


# def _update_weights(
#     self,
#     input_data: torch.Tensor,
#     bmu: Tuple[int, int],
#     t: int,
#     max_iteration: int,
# ) -> None:
#     """Update weights of the neurons.

#     Args:
#         input_data (torch.Tensor): input data tensor [batch_size, num_features]
#         bmu (Tuple[int, int]): coordinates of the winning neuron [row, col]
#         t (int): current epoch
#         max_iteration (int): total number of epochs
#     """

#     # Update learning rate and sigma at each epoch
#     learning_rate = self.lr_decay_fn(self.learning_rate, t, max_iteration)
#     sigma = self.sigma_decay_fn(self.sigma, t, max_iteration)

#     # Calculate neighborhood update (row_neurons, col_neurons)
#     neighborhood_update = self.neighborhood_fn(bmu, sigma) * learning_rate

#     # Reshape x to (1, 1, features)
#     input_data = input_data.view(1, 1, -1)

#     # Formula: w_ijk = w_ijk + h_ij * (x_ijk - w_ijk)
#     self.weights.data += torch.einsum(
#         "ij,ijk->ijk",  # 'ij' is 'neighborhood_update' (h_ij), 'ijk' is 'input_data - self.weights' (x_ijk - w_ijk)
#         neighborhood_update,  # (row_neurons, col_neurons)
#         input_data - self.weights,  # (row_neurons, col_neurons, feats)
#     )  # Update SOM weights (row_neurons, col_neurons)
