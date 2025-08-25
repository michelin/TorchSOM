"""Rectangular-specific visualization methods for Self-Organizing Maps."""

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.image import AxesImage

from torchsom.core.som import SOM
from torchsom.visualization.base_visualizer import BaseVisualizer
from torchsom.visualization.config import VisualizationConfig


class RectangularVisualizer(BaseVisualizer):
    """Specialized visualizer for rectangular topology SOMs."""

    def __init__(
        self,
        som: SOM,
        config: Optional[VisualizationConfig] = None,
    ) -> None:
        """Initialize the rectangular visualizer."""
        super().__init__(som, config, expected_topology="rectangular")

    def _customize_plot(
        self,
        ax: Axes,
        title: str,
        colorbar_label: str,
        mappable_item: Optional[AxesImage] = None,
        ticks: Optional[np.ndarray[int, Any]] = None,
        tick_labels: Optional[list[str]] = None,
    ) -> None:
        """Customize rectangular plot with proper styling.

        Args:
            ax (Axes): Matplotlib axes object to plot on
            title (str): Title of the figure to plot
            colorbar_label (str): Label for the colorbar
            mappable_item (Optional[AxesImage]): Item to plot, to adjust the colorbar values
            ticks (Optional[np.ndarray]): Ticks to plot
            tick_labels (Optional[list[str]]): Tick labels to plot
        """
        # Adjust title and axis labels
        ax.set_title(
            title,
            fontsize=self.config.fontsize["title"],
            fontweight=self.config.fontweight["title"],
            pad=10,
        )
        ax.set_xlabel(
            "Neuron Column Index",
            fontsize=self.config.fontsize["axis"],
            fontweight=self.config.fontweight["axis"],
        )
        ax.set_ylabel(
            "Neuron Row Index",
            fontsize=self.config.fontsize["axis"],
            fontweight=self.config.fontweight["axis"],
        )

        # Adjust colorbar
        if mappable_item is not None:
            cb = plt.colorbar(mappable_item, ax=ax, pad=self.config.colorbar_pad)
            cb.set_label(
                colorbar_label,
                fontsize=self.config.fontsize["axis"],
                fontweight=self.config.fontweight["axis"],
            )
            if ticks is not None:
                cb.set_ticks(ticks)
            if tick_labels is not None:
                cb.set_ticklabels(tick_labels)
            cb.ax.tick_params(labelsize=self.config.fontsize["axis"] - 2)

        # Create tick positions every 10 steps
        x_ticks = np.arange(0, self.som.y + 1, 10)
        y_ticks = np.arange(0, self.som.x + 1, 10)

        # Set tick positions and labels
        shift = 0.5
        ax.set_xticks(x_ticks - shift)
        ax.set_yticks(y_ticks - shift)
        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)

        # Add grid at the minor ticks
        ax.grid(
            which="minor",
            color="gray",
            linestyle="-",
            linewidth=0.5,
            alpha=self.config.grid_alpha,
        )
        ax.tick_params(which="minor", bottom=False, left=False)

    def plot_grid(
        self,
        map: torch.Tensor,
        title: str,
        colorbar_label: str,
        filename: str,
        save_path: Optional[Union[str, Path]] = None,
        cmap: Optional[Union[str, Colormap]] = None,
        show_values: bool = False,
        value_format: str = ".2f",
        is_component_plane: bool = False,
        **kwargs: Any,  # For compatibility with base interface # noqa: ARG002
    ) -> None:
        """Plot rectangular grid visualization.

        Args:
            map (torch.Tensor): Data to visualize [row_neurons, col_neurons]
            title (str): Plot title
            colorbar_label (str): Label for the colorbar
            filename (str): The name of the file to save
            save_path (Optional[Union[str, Path]]): Path to save the plot
            cmap (Optional[str]): Custom colormap to use
            show_values (bool): Whether to show values in cells
            value_format (str): Format string for displayed values
            is_component_plane (bool): Whether this is a component plane plot
            **kwargs: Additional arguments for compatibility
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Create a copy of the map and optionally convert 0 to NaN for visualization
        masked_map = map.clone()
        mask_zeros = kwargs.get("mask_zeros", True)
        if isinstance(masked_map, torch.Tensor):
            if mask_zeros:
                zero_mask = masked_map == 0
                masked_map[zero_mask] = float("nan")
            masked_map = masked_map.cpu().numpy()

        # Adjust the color map by setting NaN values to white
        cmap_to_use = cmap or self.config.cmap
        cmap_obj = (
            plt.cm.get_cmap(cmap_to_use).copy()
            if isinstance(cmap_to_use, str)
            else cmap_to_use.copy()
        )
        cmap_obj.set_bad(color="white")

        # Flip the data along y-axis for component planes
        if is_component_plane:
            masked_map = np.flipud(masked_map)

        # Create the image plot
        norm = kwargs.get("norm")  # Optional discrete/continuous norm
        im = ax.imshow(
            masked_map,
            cmap=cmap_obj,
            norm=norm,
            aspect="auto",
            origin="upper",  # Reverse y axis
        )

        # Customize the plot
        self._customize_plot(
            ax,
            title,
            colorbar_label,
            mappable_item=im,
            ticks=kwargs.get("ticks"),
            tick_labels=kwargs.get("tick_labels"),
        )

        # Add value annotations if requested
        if show_values:
            for i in range(masked_map.shape[0]):
                for j in range(masked_map.shape[1]):
                    if not np.isnan(masked_map[i, j]):
                        value = masked_map[i, j]
                        color = "white" if value > np.nanmean(masked_map) else "black"
                        ax.text(
                            j,
                            i,
                            f"{value:{value_format}}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=self.config.fontsize["axis"] - 4,
                            fontweight="bold",
                        )

        # Save or show the plot
        if save_path:
            self._save_plot(save_path, filename)
        else:
            plt.show()
