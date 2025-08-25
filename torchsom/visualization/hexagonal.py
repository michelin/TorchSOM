"""Hexagonal-specific visualization methods for Self-Organizing Maps."""

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure

from torchsom.core.som import SOM
from torchsom.visualization.base_visualizer import BaseVisualizer
from torchsom.visualization.config import VisualizationConfig
from torchsom.visualization.hexagonal_utils import (
    create_hexagonal_grid_patches,
    grid_to_hex_coords,
)


class HexagonalVisualizer(BaseVisualizer):
    """Specialized visualizer for hexagonal topology SOMs."""

    def __init__(
        self,
        som: SOM,
        config: Optional[VisualizationConfig] = None,
    ) -> None:
        """Initialize the hexagonal visualizer."""
        super().__init__(som, config, expected_topology="hexagonal")

    def _create_hexagonal_plot(
        self,
        map_data: torch.Tensor,
        title: str,
        colorbar_label: str,
        cmap: Optional[Union[str, Colormap]] = None,
        show_values: bool = False,
        value_format: str = ".2f",
        norm: Optional[Normalize] = None,
        ticks: Optional[np.ndarray[int, Any]] = None,
        tick_labels: Optional[list[str]] = None,
    ) -> tuple[Figure, Axes]:
        """Create a hexagonal plot with proper hexagonal patches.

        Args:
            map_data (torch.Tensor): Data to visualize [rows, cols]
            title (str): Plot title
            colorbar_label (str): Label for the colorbar
            cmap (Optional[Union[str, Colormap]]): Colormap to use
            show_values (bool): Whether to show values in hexagons
            value_format (str): Format string for displayed values
            norm (Optional[Normalize]): Normalization for the colormap
            ticks (Optional[np.ndarray]): Ticks to plot
            tick_labels (Optional[list[str]]): Tick labels to plot

        Returns:
            tuple[plt.Figure, Axes]: Figure and axes objects
        """
        # Convert to numpy if needed and handle NaN values
        if isinstance(map_data, torch.Tensor):
            map_data_np = map_data.detach().cpu().numpy()
        else:
            map_data_np = map_data.copy()

        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Set up colormap
        cmap_name_or_obj = cmap or self.config.cmap
        if isinstance(cmap_name_or_obj, str):
            cmap_obj = plt.cm.get_cmap(cmap_name_or_obj)
        else:
            cmap_obj = cmap_name_or_obj

        # Handle NaN values by setting them to white in colormap
        cmap_copy = cmap_obj.copy()
        cmap_copy.set_bad(color="white")

        # Create hexagonal patches
        patches, x_min, x_max, y_min, y_max = create_hexagonal_grid_patches(
            map_data_np,
            hex_radius=self.config.hex_radius,
            cmap=cmap_copy,
            norm=norm,
            edgecolor=self.config.hex_border_color,
            linewidth=self.config.hex_border_width,
        )

        # Add patches to the plot
        for patch in patches:
            ax.add_patch(patch)

        # Set axis limits and properties
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Invert y-axis to match grid orientation
        ax.set_aspect("equal")

        # Add title and labels
        ax.set_title(
            title,
            fontsize=self.config.fontsize["title"],
            fontweight=self.config.fontweight["title"],
            pad=20,
        )
        ax.set_xlabel(
            "Hexagonal Grid - Column Direction",
            fontsize=self.config.fontsize["axis"],
            fontweight=self.config.fontweight["axis"],
        )
        ax.set_ylabel(
            "Hexagonal Grid - Row Direction",
            fontsize=self.config.fontsize["axis"],
            fontweight=self.config.fontweight["axis"],
        )

        # Create colorbar
        valid_mask = ~np.isnan(map_data_np)
        if valid_mask.any():
            if norm is None:
                vmin = np.nanmin(map_data_np)
                vmax = np.nanmax(map_data_np)
                norm_to_use = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm_to_use = norm

            sm = plt.cm.ScalarMappable(cmap=cmap_copy, norm=norm_to_use)
            sm.set_array([])

            cb = plt.colorbar(sm, ax=ax, pad=self.config.colorbar_pad)
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

        # Add value annotations if requested
        if show_values:
            for row in range(map_data_np.shape[0]):
                for col in range(map_data_np.shape[1]):
                    if not np.isnan(map_data_np[row, col]):
                        center_x, center_y = grid_to_hex_coords(row, col)
                        value = map_data_np[row, col]
                        ax.text(
                            center_x,
                            center_y,
                            f"{value:{value_format}}",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=self.config.fontsize["axis"] - 4,
                            fontweight="bold",
                        )

        # Remove default grid and ticks for cleaner look
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        return fig, ax

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
        **kwargs: Any,  # For compatibility with base interface (ignores is_component_plane etc) # noqa: ARG002
    ) -> None:
        """Plot hexagonal grid visualization.

        Args:
            map (torch.Tensor): Data to visualize [row_neurons, col_neurons]
            title (str): Plot title
            colorbar_label (str): Label for the colorbar
            filename (str): The name of the file to save
            save_path (Optional[Union[str, Path]]): Path to save the plot
            cmap (Optional[str]): Custom colormap to use
            show_values (bool): Whether to show values in hexagons
            value_format (str): Format string for displayed values
            **kwargs: Additional arguments for compatibility (ignored)
        """
        # Optionally mask zeros to NaN for some plots. For cluster maps we keep zeros (noise label).
        masked_map = map.clone()
        mask_zeros = kwargs.get("mask_zeros", True)
        if isinstance(masked_map, torch.Tensor) and mask_zeros:
            zero_mask = masked_map == 0
            masked_map[zero_mask] = float("nan")

        # Create the hexagonal plot
        fig, ax = self._create_hexagonal_plot(
            masked_map,
            title,
            colorbar_label,
            cmap=cmap,
            show_values=show_values,
            value_format=value_format,
            norm=kwargs.get("norm"),
            ticks=kwargs.get("ticks"),
            tick_labels=kwargs.get("tick_labels"),
        )

        # Save or show the plot
        if save_path:
            self._save_plot(save_path, filename)
        else:
            plt.show()
