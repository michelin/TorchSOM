from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.image import AxesImage

from ..core import BaseSOM
from .config import VisualizationConfig


class SOMVisualizer:
    """Class for handling Self-Organizing Map visualizations."""

    def __init__(
        self,
        som: BaseSOM,
        config: Optional[VisualizationConfig] = None,
    ) -> None:
        """Initialize the SOM visualizer.

        Args:
            som (BaseSOM): Trained SOM
            config (Optional[VisualizationConfig]): Visualization configuration settings
        """

        self.som = som
        self.config = config or VisualizationConfig()
        self._setup_style()

    """
    Helper methods
    """

    def _setup_style(
        self,
    ) -> None:
        """Configure global plotting style."""

        plt.style.use("default")  # Reset matplotlib to default style
        plt.rcParams.update(
            {
                "figure.facecolor": "white",  # Background figure colore
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
        plt.rcParams["axes.prop_cycle"] = plt.cycler(
            color=colors
        )  # Each line in a multi-line plot will automatically use one of these colors in order

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
            transparent=True,  # Transparent background
        )
        plt.close()

    def _generate_hexbin_coordinates(
        self,
        map: torch.Tensor,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Generate coordinates for hexagonal grid.

        Args:
            map (torch.Tensor): SOM map to visualize. [row_neurons, col_neurons]

        Returns:
            Tuple[List[float], List[float], List[float]]: (col, row) hexbin coordinates and corresponding values. [row_neurons, col_neurons]
        """

        # Convert torch map into a numpy array
        if isinstance(map, torch.Tensor):
            map = map.detach().cpu().numpy()

        x_coords, y_coords, map_values = [], [], []
        for row in range(map.shape[0]):
            for col in range(map.shape[1]):
                x = col + (0.5 if row % 2 else 0)  # Offset to alternate rows
                y = row
                x_coords.append(x)
                y_coords.append(y)
                map_values.append(map[row, col])
        return x_coords, y_coords, map_values

    def _create_hexbin(
        self,
        ax: plt.Axes,
        x: List[float],
        y: List[float],
        values: List[float],
        gridsize: Optional[int] = None,
        log_scale: bool = False,
        cmap: Optional[str] = None,
    ) -> plt.hexbin:
        """Create hexbin plot with specified parameters.

        Args:
            ax (plt.Axes):  Matplotlib axes object to plot on
            x (List[float]): X-coordinates for hexbin
            y (List[float]): Y-coordinates for hexbin
            values (List[float]): Values to plot in hexbin
            gridsize (Optional[int], optional):  Size of hexagonal grid. If None, calculated from map dimensions. Defaults to None.
            log_scale (bool, optional): Whether to use logarithmic scale for colors. Defaults to False.
            cmap (Optional[str], optional):  Custom colormap. If None, uses default from config. Defaults to None.

        Returns:
            plt.hexbin: The created hexbin plot object
        """

        if gridsize is None:
            gridsize = self.config.hexgrid_size or int(
                min(self.som.x, self.som.y) * 1.75
            )

        # Handle NaN values by creating a mask marking indices as valid if not a NaN
        valid_mask = ~np.isnan(values)
        x_valid = np.array(x)[valid_mask]
        y_valid = np.array(y)[valid_mask]
        values_valid = np.array(values)[valid_mask]

        # Set limits for the hexbin plot
        x_min, x_max = min(x) - 0.5, max(x) + 0.5
        y_min, y_max = min(y) - 0.5, max(y) + 0.5

        return ax.hexbin(
            x_valid,
            y_valid,
            C=values_valid,
            gridsize=gridsize,  # Number of hexagons in the x-direction (and y-direction)
            cmap=cmap or self.config.cmap,
            bins="log" if log_scale else None,  # Logarithmic scale for colors
            extent=[x_min, x_max, y_min, y_max],  # Limits of the bins
            mincnt=1,  # Minimum count to consider and display a bin
            reduce_C_function=np.mean,  # Function to reduce values in each bin
        )

    def _customize_plot(
        self,
        ax: plt.Axes,
        title: str,
        colorbar_label: str,
        mappable_item: Union[AxesImage, plt.hexbin] = None,
        topology: str = "rectangular",
    ) -> None:
        """Provide a universal customization adapted to both rectangular and hexagonal settings.

        Args:
            ax (plt.Axes):  Matplotlib axes object to plot on
            title (str): Title of the figure to plot.
            colorbar_label (str): Label for the colorbar.
            mappable_item (Union[AxesImage, plt.hexbin], optional): Item to plot, to adjust the colorbar values. Defaults to None.
            topology (str): Topology of som grid. Defaults to "rectangular"
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
        cb = plt.colorbar(mappable_item, ax=ax, pad=self.config.colorbar_pad)
        cb.set_label(
            colorbar_label,
            fontsize=self.config.fontsize["axis"],
            fontweight=self.config.fontweight["axis"],
        )
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
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Set proper axis limits for the hexagonal case
        if topology == "hexagonal":
            ax.set_xlim(-1, self.som.y + 1)
            ax.set_ylim(self.som.x + 1, -1)  # Invert y-axis

    """
    Public methods
    """

    def plot_grid(
        self,
        map: torch.Tensor,
        title: str,
        colorbar_label: str,
        filename: str,
        save_path: Optional[Union[str, Path]] = None,
        log_scale: bool = False,
        cmap: Optional[str] = None,
        show_values: bool = False,
        gridsize: Optional[Tuple[int, int]] = None,
        value_format: str = ".2f",
        is_component_plane: bool = False,
    ) -> None:
        """Universal plotting function for both rectangular and hexagonal grids.

        Args:
            map (torch.Tensor): Data to visualize. [row_neurons, col_neurons]
            title (str): Plot title.
            colorbar_label (str): Label for the colorbar.
            filename (str): The name of the file to save.
            save_path (Optional[Union[str, Path]], optional): Path to save the plot. Defaults to None.
            log_scale (bool, optional): Whether to use logarithmic scale for colors. Defaults to False.
            cmap (Optional[str], optional): Custom colormap to use. Defaults to None.
            show_values (bool, optional): Whether to show values in cells. Defaults to False.
            gridsize (Optional[Tuple[int, int]], optional): Size of hexagonal grid. If None, calculated from map dimensions. Defaults to None.
            value_format (str, optional):  Format string for displayed values. Defaults to ".2f".
            is_component_plane (bool, optional): Boolean to check if current plot is a component plane. Defaults to False.
        """

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Create a copy of the map to flag values of 0 as NaN, and retrieve it as a np array
        masked_map = map.clone()
        if isinstance(masked_map, torch.Tensor):
            mask = masked_map == 0
            masked_map[mask] = float("nan")
            masked_map = masked_map.cpu().numpy()

        # Adjust the color map by setting NaN values to white
        cmap_copy = plt.cm.get_cmap(cmap or self.config.cmap).copy()
        cmap_copy.set_bad(color="white")

        if self.som.topology == "hexagonal":
            x_coords, y_coords, values = self._generate_hexbin_coordinates(
                masked_map
            )  # Create an hexbin and customize the plot
            hexbin = self._create_hexbin(
                ax,
                x_coords,
                y_coords,
                values,
                cmap=cmap_copy,
                gridsize=gridsize,
                log_scale=log_scale,
            )
            self._customize_plot(
                ax,
                title,
                colorbar_label,
                mappable_item=hexbin,
                topology=self.som.topology,
            )

        else:

            # Flip the data along y-axis for component planes
            if is_component_plane:
                masked_map = np.flipud(masked_map)
            # Create an image, add value annotations (if required) and customize the plot
            im = ax.imshow(
                masked_map,
                cmap=cmap_copy,
                aspect="auto",
                origin="upper",  # Reverse y axis
            )
            self._customize_plot(
                ax,
                title,
                colorbar_label,
                mappable_item=im,
                topology=self.som.topology,
            )
            if show_values:
                for i in range(masked_map.shape[0]):
                    for j in range(masked_map.shape[1]):
                        if not np.isnan(masked_map[i, j]):
                            value = masked_map[i, j]
                            color = (
                                "white" if value > np.nanmean(masked_map) else "black"
                            )
                            ax.text(
                                j,
                                i,
                                f"{value:{value_format}}",
                                ha="center",
                                va="center",
                                color=color,
                            )

        if save_path:
            self._save_plot(save_path, name=f"{filename}")
        else:
            plt.show()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def plot_training_errors(
        self,
        quantization_errors: List[float],
        topographic_errors: List[float],
        fig_name: Optional[str] = "training_errors",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot training errors over epochs.

        Args:
            quantization_errors (List[float]): List of quantization errors [epochs]
            topographic_errors (List[float]): List of topographic errors [epochs]
            fig_name (Optional[str], optional): The name of the file to save.. Defaults to "training_errors".
            save_path (Optional[Union[str, Path]], optional): Optional path to save the visualization figure. Defaults to None.
        """

        # Ensure tensors are moved to CPU before plotting
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
        ax1.set_xlabel(
            "Epoch",
            fontsize=self.config.fontsize["axis"],
            fontweight=self.config.fontweight["axis"],
        )
        ax1.set_ylabel(
            "Value",
            fontsize=self.config.fontsize["axis"],
            fontweight=self.config.fontweight["axis"],
        )
        ax1.grid(True, alpha=self.config.grid_alpha)

        # Plot topographic errors
        ax2.plot(epochs, topographic_errors, color="orange", linewidth=2)
        ax2.set_title(
            "Topographic Error",
            fontsize=self.config.fontsize["title"],
            fontweight=self.config.fontweight["title"],
        )
        ax2.set_xlabel(
            "Epoch",
            fontsize=self.config.fontsize["axis"],
            fontweight=self.config.fontweight["axis"],
        )
        ax2.set_ylabel(
            "Ratio (%)",
            fontsize=self.config.fontsize["axis"],
            fontweight=self.config.fontweight["axis"],
        )
        ax2.grid(True, alpha=self.config.grid_alpha)

        if save_path:
            self._save_plot(save_path=save_path, name=f"{fig_name}")
        else:
            plt.show()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Additional visualization methods such as plot_distance_map, plot_hit_map, etc.
    # would be implemented here following the same pattern as in the original code
