"""Base visualizer class with common functionality for all SOM topologies."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import torch

from torchsom.core import BaseSOM
from torchsom.visualization.config import VisualizationConfig


class BaseVisualizer(ABC):
    """Abstract base class for SOM visualizers with common functionality."""

    def __init__(
        self,
        som: BaseSOM,
        config: Optional[VisualizationConfig] = None,
        expected_topology: str = None,
    ) -> None:
        """Initialize the base visualizer.

        Args:
            som (BaseSOM): Trained SOM
            config (Optional[VisualizationConfig]): Visualization configuration settings
            expected_topology (str): Expected topology for validation
        """
        self.som = som
        self.config = config or VisualizationConfig()
        if expected_topology and self.som.topology != expected_topology:
            raise ValueError(
                f"{self.__class__.__name__} requires SOM with {expected_topology} topology"
            )

    def _prepare_save_path(
        self,
        save_path: Union[str, Path],
    ) -> Path:
        """Prepare directory for saving visualizations.

        Args:
            save_path (Union[str, Path]): The path to save the visualization.

        Returns:
            Path: The path to save the visualization.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def _save_plot(
        self,
        save_path: Union[str, Path],
        name: str,
    ) -> None:
        """Save plot with specified configuration.

        Args:
            save_path (Union[str, Path]): The path to save the visualization.
            name (str): The name of the file to save.
        """
        save_path = self._prepare_save_path(save_path=save_path)
        plt.savefig(
            save_path / f"{name}.{self.config.save_format}",
            dpi=self.config.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=True,
        )
        plt.close()

    @abstractmethod
    def plot_grid(
        self,
        map: torch.Tensor,
        title: str,
        colorbar_label: str,
        filename: str,
        save_path: Optional[Union[str, Path]] = None,
        cmap: Optional[str] = None,
        show_values: bool = False,
        value_format: str = ".2f",
        **kwargs: Any,
    ) -> None:
        """Plot grid visualization - must be implemented by subclasses.

        Args:
            map (torch.Tensor): Data to visualize [row_neurons, col_neurons]
            title (str): Plot title
            colorbar_label (str): Label for the colorbar
            filename (str): The name of the file to save
            save_path (Optional[Union[str, Path]]): Path to save the plot
            cmap (Optional[str]): Custom colormap to use
            show_values (bool): Whether to show values in cells
            value_format (str): Format string for displayed values
            **kwargs: Additional arguments for topology-specific needs
        """
        pass

    def plot_distance_map(
        self,
        fig_name: str = "distance_map",
        save_path: Optional[Union[str, Path]] = None,
        distance_metric: Optional[str] = None,
        neighborhood_order: Optional[int] = None,
        scaling: str = "sum",
    ) -> None:
        """Plot the distance map (U-Matrix).

        Args:
            fig_name (str): The name of the file to save
            save_path (Optional[Union[str, Path]]): Path to save the visualization
            distance_metric (Optional[str]): Distance function name
            neighborhood_order (Optional[int]): Neighbor order to consider
            scaling (str): 'sum' or 'mean' aggregation
        """
        distance_map = self.som.build_map(
            "distance",
            distance_metric=distance_metric,
            neighborhood_order=neighborhood_order or self.som.neighborhood_order,
            scaling=scaling,
        )
        self.plot_grid(
            map=distance_map,
            title=f"U-Matrix (Distance Map) - Order {neighborhood_order or self.som.neighborhood_order}",
            colorbar_label=f"{distance_metric or self.som.distance_fn_name} distance",
            filename=fig_name,
            save_path=save_path,
        )

    def plot_hit_map(
        self,
        data: torch.Tensor,
        fig_name: str = "hit_map",
        save_path: Optional[Union[str, Path]] = None,
        batch_size: int = 1024,
    ) -> None:
        """Plot hit map.

        Args:
            data (torch.Tensor): Input data tensor [batch_size, n_features]
            fig_name (str): The name of the file to save
            save_path (Optional[Union[str, Path]]): Path to save the visualization
            batch_size (int): Batch processing size
        """
        hit_map = self.som.build_map("hit", data=data, batch_size=batch_size)
        self.plot_grid(
            map=hit_map,
            title="Hit Map",
            colorbar_label="Number of Hits",
            filename=fig_name,
            save_path=save_path,
        )

    def plot_classification_map(
        self,
        bmus_data_map: dict[tuple[int, int], list[int]],
        data: torch.Tensor,
        target: torch.Tensor,
        fig_name: str = "classification_map",
        save_path: Optional[Union[str, Path]] = None,
        neighborhood_order: Optional[int] = None,
    ) -> None:
        """Plot classification map.

        Args:
            bmus_data_map (dict[tuple[int, int], list[int]]): Pre-computed BMU to data indices mapping
            data (torch.Tensor): Input data tensor [batch_size, n_features]
            target (torch.Tensor): Labels tensor for data points [batch_size]
            fig_name (str): The name of the file to save
            save_path (Optional[Union[str, Path]]): Path to save the visualization
            neighborhood_order (Optional[int]): Neighborhood order for tie-breaking
        """
        classification_map = self.som.build_map(
            "classification",
            data=data,
            target=target,
            neighborhood_order=neighborhood_order or self.som.neighborhood_order,
            bmus_data_map=bmus_data_map,
        )
        self.plot_grid(
            map=classification_map,
            title="Classification Map",
            colorbar_label="Most Frequent Encoded Label",
            filename=fig_name,
            save_path=save_path,
        )

    def plot_metric_map(
        self,
        bmus_data_map: dict[tuple[int, int], list[int]],
        data: torch.Tensor,
        target: torch.Tensor,
        reduction_parameter: str = "mean",
        fig_name: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot target metric map.

        Args:
            bmus_data_map (dict[tuple[int, int], list[int]]): Pre-computed BMU to data indices mapping
            data (torch.Tensor): Input data tensor [batch_size, n_features]
            target (torch.Tensor): Labels tensor for data points [batch_size]
            reduction_parameter (str): Calculation to apply ('mean' or 'std')
            fig_name (Optional[str]): The name of the file to save
            save_path (Optional[Union[str, Path]]): Path to save the visualization
        """
        metric_map = self.som.build_map(
            "metric",
            data=data,
            target=target,
            reduction_parameter=reduction_parameter,
            bmus_data_map=bmus_data_map,
        )
        title = (
            "Map of Mean Target Value"
            if reduction_parameter == "mean"
            else "Map of Standard Deviation of Target Values"
        )
        fig_name = fig_name or f"{reduction_parameter}_target_map"
        self.plot_grid(
            map=metric_map,
            title=title,
            colorbar_label=title,
            filename=fig_name,
            save_path=save_path,
        )

    def plot_score_map(
        self,
        bmus_data_map: dict[tuple[int, int], list[int]],
        target: torch.Tensor,
        total_samples: int,
        fig_name: str = "score_map",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot neuron representativeness score map.

        Args:
            bmus_data_map (dict[tuple[int, int], list[int]]): Pre-computed BMU to data indices mapping
            target (torch.Tensor): Labels tensor for data points [batch_size]
            total_samples (int): Total number of samples
            fig_name (str): The name of the file to save
            save_path (Optional[Union[str, Path]]): Path to save the visualization
        """
        score_map = self.som.build_map(
            "score",
            bmus_data_map=bmus_data_map,
            target=target,
            total_samples=total_samples,
        )
        self.plot_grid(
            map=score_map,
            title="Neuron Representativeness Map",
            colorbar_label="Relevance Score (lower = better)",
            filename=fig_name,
            save_path=save_path,
        )

    def plot_rank_map(
        self,
        bmus_data_map: dict[tuple[int, int], list[int]],
        target: torch.Tensor,
        fig_name: str = "rank_map",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot ranked neurons map.

        Args:
            bmus_data_map (dict[tuple[int, int], list[int]]): Pre-computed BMU to data indices mapping
            target (torch.Tensor): Labels tensor for data points [batch_size]
            fig_name (str): The name of the file to save
            save_path (Optional[Union[str, Path]]): Path to save the visualization
        """
        rank_map = self.som.build_map(
            "rank",
            target=target,
            bmus_data_map=bmus_data_map,
        )
        self.plot_grid(
            map=rank_map,
            title="Neuron Map Ranked by Output Std",
            colorbar_label="Rank (lower std = higher rank)",
            filename=fig_name,
            save_path=save_path,
            show_values=True,
            value_format=".0f",
        )

    def plot_component_planes(
        self,
        component_names: Optional[list[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot component planes.

        Args:
            component_names (Optional[list[str]]): Names for each component/feature
            save_path (Optional[Union[str, Path]]): Path to save the visualization
        """
        n_components = self.som.weights.shape[-1]
        component_names = component_names or [
            f"Component_{i+1}" for i in range(n_components)
        ]
        for i, name in enumerate(component_names):
            component_weights = self.som.weights[:, :, i].cpu()
            self.plot_grid(
                map=component_weights,
                title=f"Component Plane {name}",
                colorbar_label=f"{name} Weight Values",
                filename=name,
                save_path=f"{save_path}/component_planes" if save_path else None,
                is_component_plane=True,  # This will be ignored by hexagonal but used by rectangular
            )

    def plot_training_errors(
        self,
        quantization_errors: list[float],
        topographic_errors: list[float],
        fig_name: str = "training_errors",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot training errors over epochs.

        Args:
            quantization_errors (list[float]): List of quantization errors
            topographic_errors (list[float]): List of topographic errors
            fig_name (str): The name of the file to save
            save_path (Optional[Union[str, Path]]): Path to save the visualization
        """
        # Ensure data is on CPU
        if isinstance(quantization_errors, torch.Tensor):
            quantization_errors = quantization_errors.cpu().numpy()
        if isinstance(topographic_errors, torch.Tensor):
            topographic_errors = topographic_errors.cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=self.config.figsize, gridspec_kw={"hspace": 0.3}
        )

        epochs = range(len(quantization_errors))

        # Plot quantization errors
        ax1.plot(epochs, quantization_errors, color="blue", linewidth=2)
        ax1.set_title(
            "Quantization Error",
            fontsize=self.config.fontsize["title"],
            fontweight=self.config.fontweight["title"],
        )
        ax1.set_xlabel("Epoch", fontsize=self.config.fontsize["axis"])
        ax1.set_ylabel("Value", fontsize=self.config.fontsize["axis"])
        ax1.grid(True, alpha=0.3)

        # Plot topographic errors
        ax2.plot(epochs, topographic_errors, color="orange", linewidth=2)
        ax2.set_title(
            "Topographic Error",
            fontsize=self.config.fontsize["title"],
            fontweight=self.config.fontweight["title"],
        )
        ax2.set_xlabel("Epoch", fontsize=self.config.fontsize["axis"])
        ax2.set_ylabel("Ratio (%)", fontsize=self.config.fontsize["axis"])
        ax2.grid(True, alpha=0.3)

        if save_path:
            self._save_plot(save_path, fig_name)
        else:
            plt.show()
