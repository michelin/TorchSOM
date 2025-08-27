"""Clustering algorithms for SOM analysis using scikit-learn."""

import warnings
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from hdbscan import HDBSCAN, prediction
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

if TYPE_CHECKING:
    from torchsom.core.base_som import BaseSOM


def cluster_kmeans(
    data: torch.Tensor,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """K-means clustering using scikit-learn.

    Args:
        data (torch.Tensor): Input data [n_samples, n_features]
        n_clusters (Optional[int]): Number of clusters. If None, uses elbow method
        random_state (int): Random seed for reproducibility
        **kwargs: Additional arguments for KMeans

    Returns:
        dict[str, Any]: Clustering result with labels, centers, and metadata
    """
    # Convert to numpy for sklearn
    data_np = data.detach().cpu().numpy()
    n_samples = data_np.shape[0]

    # Determine optimal k if not provided
    if n_clusters is None:
        n_clusters = _determine_optimal_k_elbow(data_np, random_state=random_state)

    # Ensure n_clusters is reasonable
    n_clusters = min(n_clusters, max(1, n_samples - 1))

    # Apply K-means
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10, **kwargs
    )
    labels = kmeans.fit_predict(data_np)

    # Convert to 1-indexed labeling (1 to n instead of 0 to n-1)
    labels = labels + 1

    # Convert back to tensors on original device
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=data.device)
    centers_tensor = torch.tensor(
        kmeans.cluster_centers_, dtype=data.dtype, device=data.device
    )

    return {
        "labels": labels_tensor,
        "centers": centers_tensor,
        "n_clusters": n_clusters,
        "method": "kmeans",
        "inertia": kmeans.inertia_,
    }


def cluster_gmm(
    data: torch.Tensor,
    n_components: Optional[int] = None,
    random_state: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Gaussian Mixture Model clustering using scikit-learn.

    Args:
        data (torch.Tensor): Input data [n_samples, n_features]
        n_components (Optional[int]): Number of components. If None, uses BIC selection
        random_state (int): Random seed for reproducibility
        **kwargs: Additional arguments for GaussianMixture

    Returns:
        dict[str, Any]: Clustering result with labels, centers, and metadata
    """
    # Convert to numpy for sklearn
    data_np = data.detach().cpu().numpy()
    n_samples = data_np.shape[0]

    # Determine optimal number of components if not provided
    if n_components is None:
        n_components = _determine_optimal_components_bic(
            data_np, random_state=random_state
        )

    # Ensure n_components is reasonable
    n_components = min(n_components, max(1, n_samples - 1))

    # Apply GMM
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        init_params="k-means++",
        **kwargs,
    )
    gmm.fit(data_np)
    labels = gmm.predict(data_np)

    # Convert to 1-indexed labeling (1 to n instead of 0 to n-1)
    labels = labels + 1

    # Convert back to tensors on original device
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=data.device)
    centers_tensor = torch.tensor(gmm.means_, dtype=data.dtype, device=data.device)

    return {
        "labels": labels_tensor,
        "centers": centers_tensor,
        "n_clusters": n_components,
        "method": "gmm",
        "bic": gmm.bic(data_np),
        "aic": gmm.aic(data_np),
    }


def cluster_hdbscan(
    data: torch.Tensor,
    min_cluster_size: Optional[int] = None,
    **kwargs: Any,
) -> dict[str, Any]:  # pragma: no cover
    """HDBSCAN clustering using scikit-learn.

    Args:
        data (torch.Tensor): Input data [n_samples, n_features]
        min_cluster_size (Optional[int]): Minimum size of clusters
        **kwargs: Additional arguments for HDBSCAN

    Returns:
        dict[str, Any]: Clustering result with labels, centers, and metadata
    """
    # Convert to numpy for sklearn
    data_np = data.detach().cpu().numpy()
    n_samples = data_np.shape[0]

    # Set smart default for min_cluster_size
    if min_cluster_size is None:
        min_cluster_size = max(5, n_samples // 20)

    # Apply HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        prediction_data=True,
        **kwargs,
    )
    # labels = clusterer.fit_predict(data_np)
    clusterer.fit(data_np)
    labels, strengths = prediction.approximate_predict(clusterer, data_np)

    # Calculate cluster centers (excluding noise points)
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels != -1]
    n_clusters = len(valid_labels)

    if n_clusters == 0:
        warnings.warn(
            "HDBSCAN found no clusters. All points classified as noise.", stacklevel=2
        )
        # Create a single cluster with all points (1-indexed)
        labels = np.ones_like(labels)
        n_clusters = 1
        valid_labels = [1]
    else:
        # Convert non-noise labels to contiguous 1-indexed labels
        # Keep noise points as -1, but remap clusters to 1, 2, 3, ... (contiguous)
        labels_1indexed = labels.copy()

        # Create mapping from old labels to new contiguous labels starting from 1
        label_mapping = {}
        for i, old_label in enumerate(sorted(valid_labels)):
            label_mapping[old_label] = i + 1

        # Apply mapping to all non-noise points
        for old_label, new_label in label_mapping.items():
            labels_1indexed[labels == old_label] = new_label

        labels = labels_1indexed
        valid_labels = list(range(1, len(valid_labels) + 1))  # [1, 2, 3, ...]

    # Calculate centers for non-noise clusters
    centers_list = []
    for label in sorted(valid_labels):
        cluster_mask = labels == label
        if cluster_mask.sum() > 0:
            center = data_np[cluster_mask].mean(axis=0)
            centers_list.append(center)

    centers = (
        np.array(centers_list) if centers_list else data_np.mean(axis=0, keepdims=True)
    )

    # Convert back to tensors on original device
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=data.device)
    centers_tensor = torch.tensor(centers, dtype=data.dtype, device=data.device)

    return {
        "labels": labels_tensor,
        "centers": centers_tensor,
        "n_clusters": n_clusters,
        "method": "hdbscan",
        "noise_points": (labels == -1).sum(),
    }


def _determine_optimal_k_elbow(
    data_np: np.ndarray[Any, np.dtype[np.floating[Any]]],
    max_k: int = 10,
    random_state: int = 42,
) -> int:
    """Determine optimal number of clusters using elbow method."""
    n_samples = data_np.shape[0]
    max_k = min(max_k, max(2, int(np.sqrt(n_samples))))

    if max_k < 2:
        return 2

    # Calculate inertias for different k values
    inertias = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=3)
        kmeans.fit(data_np)
        inertias.append(kmeans.inertia_)

    # Find elbow using second derivative method
    if len(inertias) < 3:
        return k_range[0]

    # Calculate second derivatives
    second_derivatives = []
    for i in range(1, len(inertias) - 1):
        second_deriv = inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
        second_derivatives.append(second_deriv)

    # Find the point with maximum second derivative (sharpest bend)
    if second_derivatives:
        elbow_idx = np.argmax(second_derivatives) + 1
        optimal_k = k_range[elbow_idx]
    else:
        optimal_k = k_range[len(k_range) // 2]

    return optimal_k


def _determine_optimal_components_bic(
    data_np: np.ndarray[Any, np.dtype[np.floating[Any]]],
    max_components: int = 10,
    random_state: int = 42,
) -> int:
    """Determine optimal number of components using BIC."""
    n_samples = data_np.shape[0]
    max_components = min(max_components, max(2, int(np.sqrt(n_samples))))

    if max_components < 2:
        return 2

    # Calculate BIC for different numbers of components
    bic_scores = []
    component_range = range(1, max_components + 1)

    for n_comp in component_range:
        try:
            gmm = GaussianMixture(
                n_components=n_comp,
                random_state=random_state,
                init_params="k-means++",
            )
            gmm.fit(data_np)
            bic_scores.append(gmm.bic(data_np))
        except Exception:
            bic_scores.append(float("inf"))

    # Select component with lowest BIC
    optimal_components = component_range[np.argmin(bic_scores)]
    return optimal_components


def extract_clustering_features(
    som: "BaseSOM",
    feature_space: str,
) -> torch.Tensor:
    """Extract features for clustering based on feature space specification.

    Args:
        som (SOM): SOM instance
        feature_space (str): "weights", "positions", or "combined"

    Returns:
        torch.Tensor: Features [n_neurons, n_features]
    """
    if feature_space == "weights":
        return som.weights.view(-1, som.num_features)

    elif feature_space == "positions":
        return torch.stack([som.xx.flatten(), som.yy.flatten()], dim=1)

    elif feature_space == "combined":
        weights_flat = som.weights.view(-1, som.num_features)
        positions = torch.stack([som.xx.flatten(), som.yy.flatten()], dim=1)
        return torch.cat([weights_flat, positions], dim=1)

    else:
        raise ValueError(f"Unsupported feature space: {feature_space}")


def cluster_data(
    data: torch.Tensor,
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Main clustering function that dispatches to specific algorithms.

    Args:
        data (torch.Tensor): Input data [n_samples, n_features]
        method (str): Clustering method ("kmeans", "gmm", "hdbscan")
        n_clusters (Optional[int]): Number of clusters. If None, uses automatic selection
        **kwargs: Additional arguments for specific clustering methods

    Returns:
        dict[str, Any]: Clustering result

    Raises:
        ValueError: If an unsupported clustering method is specified
    """
    if data.numel() == 0:
        raise ValueError("Cannot cluster empty data")

    if data.dim() != 2:
        raise ValueError("Data must be 2D tensor [n_samples, n_features]")

    if data.shape[0] < 2:
        raise ValueError("Need at least 2 samples for clustering")

    if method == "kmeans":
        return cluster_kmeans(data, n_clusters=n_clusters, **kwargs)
    elif method == "gmm":
        return cluster_gmm(data, n_components=n_clusters, **kwargs)
    elif method == "hdbscan":
        # if n_clusters is not None:
        #     warnings.warn(
        #         "n_clusters parameter is ignored for HDBSCAN. "
        #         "Use min_cluster_size to influence cluster formation.",
        #         stacklevel=2,
        #     )
        return cluster_hdbscan(data, **kwargs)
    else:
        raise ValueError(
            f"Unsupported clustering method: {method}. "
            "Supported methods: 'kmeans', 'gmm', 'hdbscan'"
        )
