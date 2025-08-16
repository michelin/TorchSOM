"""Hexagonal-specific visualization methods for Self-Organizing Maps."""

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
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
    ) -> tuple[Figure, Axes]:
        """Create a hexagonal plot with proper hexagonal patches.

        Args:
            map_data (torch.Tensor): Data to visualize [rows, cols]
            title (str): Plot title
            colorbar_label (str): Label for the colorbar
            cmap (Optional[Union[str, Colormap]]): Colormap to use
            show_values (bool): Whether to show values in hexagons
            value_format (str): Format string for displayed values

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
        cmap_name = cmap or self.config.cmap
        if isinstance(cmap_name, str):
            cmap_obj = plt.cm.get_cmap(cmap_name)
        else:
            cmap_obj = cmap_name

        # Handle NaN values by setting them to white in colormap
        cmap_copy = cmap_obj.copy()
        cmap_copy.set_bad(color="white")

        # Create hexagonal patches
        patches, x_min, x_max, y_min, y_max = create_hexagonal_grid_patches(
            map_data_np,
            hex_radius=self.config.hex_radius,
            cmap_name=cmap_name if isinstance(cmap_name, str) else "viridis",
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
            vmin = np.nanmin(map_data_np)
            vmax = np.nanmax(map_data_np)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap_copy, norm=norm)
            sm.set_array([])

            cb = plt.colorbar(sm, ax=ax, pad=self.config.colorbar_pad)
            cb.set_label(
                colorbar_label,
                fontsize=self.config.fontsize["axis"],
                fontweight=self.config.fontweight["axis"],
            )
            cb.ax.tick_params(labelsize=self.config.fontsize["axis"] - 2)

        # Add value annotations if requested
        if show_values:
            for row in range(map_data_np.shape[0]):
                for col in range(map_data_np.shape[1]):
                    if not np.isnan(map_data_np[row, col]):
                        center_x, center_y = grid_to_hex_coords(row, col)
                        value = map_data_np[row, col]

                        # Choose text color based on background
                        if valid_mask.any():
                            normalized_value = (
                                (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                            )
                            text_color = "white" if normalized_value > 0.5 else "black"
                        else:
                            text_color = "black"

                        ax.text(
                            center_x,
                            center_y,
                            f"{value:{value_format}}",
                            ha="center",
                            va="center",
                            color=text_color,
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
        cmap: Optional[str] = None,
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
        # Create masked map (convert 0 to NaN for visualization)
        masked_map = map.clone()
        if isinstance(masked_map, torch.Tensor):
            mask = masked_map == 0
            masked_map[mask] = float("nan")

        # Create the hexagonal plot
        fig, ax = self._create_hexagonal_plot(
            masked_map,
            title,
            colorbar_label,
            cmap=cmap,
            show_values=show_values,
            value_format=value_format,
        )

        # Save or show the plot
        if save_path:
            self._save_plot(save_path, filename)
        else:
            plt.show()

        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
