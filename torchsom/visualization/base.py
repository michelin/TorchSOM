"""Base class for all visualization methods."""

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import torch

from torchsom.core import BaseSOM
from torchsom.visualization.clustering import ClusteringVisualizer
from torchsom.visualization.config import VisualizationConfig
from torchsom.visualization.hexagonal import HexagonalVisualizer
from torchsom.visualization.rectangular import RectangularVisualizer


class SOMVisualizer:
    """Factory class for handling Self-Organizing Map visualizations.

    This class acts as a factory that creates the appropriate topology-specific
    visualizer (HexagonalVisualizer or RectangularVisualizer) based on the SOM's topology.
    """

    def __init__(
        self,
        som: BaseSOM,
        config: Optional[VisualizationConfig] = None,
    ) -> None:
        """Initialize the SOM visualizer factory.

        Args:
            som (BaseSOM): Trained SOM
            config (Optional[VisualizationConfig]): Visualization configuration settings
        """
        self.som = som
        self.config = config or VisualizationConfig()
        self._setup_style()

        # Create the appropriate topology-specific visualizer
        if som.topology == "hexagonal":
            self._visualizer = HexagonalVisualizer(som, config)
        elif som.topology == "rectangular":
            self._visualizer = RectangularVisualizer(som, config)
        else:
            raise ValueError(f"Unsupported topology: {som.topology}")

        # Create clustering visualizer
        self._clustering_visualizer = ClusteringVisualizer(som, config)

    def _setup_style(self) -> None:
        """Configure global plotting style."""
        plt.style.use("default")  # Reset matplotlib to default style
        plt.rcParams.update(
            {
                "figure.facecolor": "white",  # Background figure color
                "axes.facecolor": "white",  # Background axes color
                "axes.grid": True,  # Show grid
                "grid.alpha": self.config.grid_alpha,  # Grid transparency
                "axes.labelsize": self.config.fontsize["axis"],  # Axis label size
                "axes.titlesize": self.config.fontsize["title"],  # Title size
                "xtick.labelsize": self.config.fontsize["axis"]
                - 2,  # X-axis tick label size
                "ytick.labelsize": self.config.fontsize["axis"]
                - 2,  # Y-axis tick label size
                "axes.spines.top": True,  # Show top spine (border of the plot)
                "axes.spines.right": True,
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.axisbelow": True,  # Show grid below the plot
                "lines.linewidth": 1.5,  # Line thickness for plots
                "grid.linestyle": "--",  # Grid line style
                "grid.color": "gray",  # Grid line color
            }
        )
        colors = [
            "#4477AA",  # Dark blue
            "#66CCEE",  # Light blue
            "#228833",  # Green
            "#CCBB44",  # Yellow
            "#EE6677",  # Red
            "#AA3377",  # Purple
            "#BBBBBB",  # Gray
        ]
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)

    # Delegate all visualization methods to the topology-specific visualizer
    def plot_grid(self, *args: Any, **kwargs: Any) -> None:
        """Plot grid visualization using topology-specific visualizer."""
        return self._visualizer.plot_grid(*args, **kwargs)

    def plot_training_errors(self, *args: Any, **kwargs: Any) -> None:
        """Plot training errors using topology-specific visualizer."""
        return self._visualizer.plot_training_errors(*args, **kwargs)

    def plot_distance_map(self, *args: Any, **kwargs: Any) -> None:
        """Plot distance map using topology-specific visualizer."""
        return self._visualizer.plot_distance_map(*args, **kwargs)

    def plot_classification_map(self, *args: Any, **kwargs: Any) -> None:
        """Plot classification map using topology-specific visualizer."""
        return self._visualizer.plot_classification_map(*args, **kwargs)

    def plot_hit_map(self, *args: Any, **kwargs: Any) -> None:
        """Plot hit map using topology-specific visualizer."""
        return self._visualizer.plot_hit_map(*args, **kwargs)

    def plot_metric_map(self, *args: Any, **kwargs: Any) -> None:
        """Plot metric map using topology-specific visualizer."""
        return self._visualizer.plot_metric_map(*args, **kwargs)

    def plot_score_map(self, *args: Any, **kwargs: Any) -> None:
        """Plot score map using topology-specific visualizer."""
        return self._visualizer.plot_score_map(*args, **kwargs)

    def plot_rank_map(self, *args: Any, **kwargs: Any) -> None:
        """Plot rank map using topology-specific visualizer."""
        return self._visualizer.plot_rank_map(*args, **kwargs)

    def plot_component_planes(self, *args: Any, **kwargs: Any) -> None:
        """Plot component planes using topology-specific visualizer."""
        return self._visualizer.plot_component_planes(*args, **kwargs)

    # Clustering visualization methods
    def plot_cluster_map(self, *args: Any, **kwargs: Any) -> None:
        """Plot clustering results overlaid on SOM grid."""
        return self._clustering_visualizer.plot_cluster_map(*args, **kwargs)

    def plot_silhouette_analysis(self, *args: Any, **kwargs: Any) -> None:
        """Plot silhouette analysis for clustering results."""
        return self._clustering_visualizer.plot_silhouette_analysis(*args, **kwargs)

    def plot_cluster_quality_comparison(self, *args: Any, **kwargs: Any) -> None:
        """Compare clustering quality metrics across different methods."""
        return self._clustering_visualizer.plot_cluster_quality_comparison(
            *args, **kwargs
        )

    def plot_elbow_analysis(self, *args: Any, **kwargs: Any) -> None:
        """Plot elbow analysis for optimal K selection in K-means."""
        return self._clustering_visualizer.plot_elbow_analysis(*args, **kwargs)

    def plot_clustering_comparison_grid(self, *args: Any, **kwargs: Any) -> None:
        """Plot a grid comparing different clustering methods and feature spaces."""
        return self._clustering_visualizer.plot_clustering_comparison_grid(
            *args, **kwargs
        )

    def plot_all(
        self,
        quantization_errors: list[float],
        topographic_errors: list[float],
        data: torch.Tensor,
        target: torch.Tensor,
        component_names: Optional[list[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        training_errors: bool = True,
        distance_map: bool = True,
        hit_map: bool = True,
        score_map: bool = True,
        rank_map: bool = True,
        metric_map: bool = True,
        component_planes: bool = True,
    ) -> None:
        """Plot all visualizations using topology-specific visualizer.

        Args:
            quantization_errors (list[float]): List of quantization errors [epochs]
            topographic_errors (list[float]): List of topographic errors [epochs]
            data (torch.Tensor): Input data tensor [batch_size, n_features]
            target (torch.Tensor): Labels tensor for data points [batch_size]
            component_names (Optional[list[str]]): Names for each component/feature
            save_path (Optional[Union[str, Path]]): Path to save visualizations
            training_errors (bool): Whether to plot training learning curves
            distance_map (bool): Whether to plot distance map
            hit_map (bool): Whether to plot hit map
            score_map (bool): Whether to plot score map
            rank_map (bool): Whether to plot rank map
            metric_map (bool): Whether to plot metric map
            component_planes (bool): Whether to plot component planes
        """
        if training_errors:
            self._visualizer.plot_training_errors(
                quantization_errors=quantization_errors,
                topographic_errors=topographic_errors,
                save_path=save_path,
            )
        if distance_map:
            self._visualizer.plot_distance_map(save_path=save_path)
        if hit_map:
            self._visualizer.plot_hit_map(data, save_path=save_path)
        if metric_map:
            self._visualizer.plot_metric_map(
                data, target, reduction_parameter="mean", save_path=save_path
            )
            self._visualizer.plot_metric_map(
                data, target, reduction_parameter="std", save_path=save_path
            )
        if score_map:
            self._visualizer.plot_score_map(data, target, save_path=save_path)
        if rank_map:
            self._visualizer.plot_rank_map(data, target, save_path=save_path)
        if component_planes:
            self._visualizer.plot_component_planes(
                component_names=component_names, save_path=save_path
            )
