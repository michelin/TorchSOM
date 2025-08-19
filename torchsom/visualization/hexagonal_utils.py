"""Utility functions for hexagonal grid visualization."""

import math
from typing import Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import RegularPolygon

# NOTE: Coordinate conversion and distance functions moved to torchsom.utils.hexagonal_coordinates to eliminate duplication.


def grid_to_hex_coords(
    row: int,
    col: int,
) -> tuple[float, float]:
    """Convert grid indices to hexagonal visualization coordinates.

    Uses even-r offset coordinate system:
    - Even rows (0, 2, 4...) are not shifted horizontally
    - Odd rows (1, 3, 5...) are shifted right by 0.5 units
    - Vertical spacing uses proper hexagonal ratio

    Args:
        row (int): Grid row index
        col (int): Grid column index

    Returns:
        tuple[float, float]: (x, y) coordinates for hexagon center
    """
    # Use the normalized coordinate system, then scale for visualization
    x_norm = col + (0.5 if row % 2 == 1 else 0.0)
    y_norm = row * (math.sqrt(3) / 2)

    return x_norm, y_norm


def calculate_hex_dimensions(
    x_dim: int,
    y_dim: int,
    hex_radius: float = 0.4,
) -> tuple[float, float, float, float]:
    """Calculate the bounding box for the hexagonal grid.

    Args:
        x_dim (int): Number of rows in the grid
        y_dim (int): Number of columns in the grid
        hex_radius (float): Radius of hexagonal cells

    Returns:
        Tuple[float, float, float, float]: (x_min, x_max, y_min, y_max)
    """
    # Calculate all hexagon centers
    x_coords = []
    y_coords = []

    for row in range(x_dim):
        for col in range(y_dim):
            x, y = grid_to_hex_coords(row, col)
            x_coords.append(x)
            y_coords.append(y)

    # Add padding based on hexagon radius
    padding = hex_radius * 1.2
    x_min = min(x_coords) - padding
    x_max = max(x_coords) + padding
    y_min = min(y_coords) - padding
    y_max = max(y_coords) + padding

    return x_min, x_max, y_min, y_max


def create_hexagon_patch(
    center_x: float,
    center_y: float,
    radius: float = 0.4,
    facecolor: str = "white",
    edgecolor: str = "black",
    linewidth: float = 0.5,
) -> RegularPolygon:
    """Create a hexagonal patch for matplotlib.

    Args:
        center_x (float): X coordinate of hexagon center
        center_y (float): Y coordinate of hexagon center
        radius (float): Radius of the hexagon
        facecolor (str): Fill color of the hexagon
        edgecolor (str): Border color of the hexagon
        linewidth (float): Width of the border

    Returns:
        RegularPolygon: Matplotlib hexagon patch
    """
    return RegularPolygon(
        (center_x, center_y),
        6,  # number of sides (hexagon)
        radius=radius,
        orientation=0,  # flat-top orientation = math.pi / 2 and flat-right orientation = 0
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )


def create_hexagonal_grid_patches(
    map_data: torch.Tensor,
    hex_radius: float = 0.4,
    cmap: Union[Colormap, None] = None,
    norm: Union[Normalize, None] = None,
    edgecolor: str = "white",
    linewidth: float = 0.5,
) -> tuple[list[RegularPolygon], float, float, float, float]:
    """Create hexagonal patches for the entire grid.

    Args:
        map_data (torch.Tensor): Data to visualize [rows, cols]
        hex_radius (float): Radius of hexagonal cells
        cmap_name (str): Name of the colormap to use
        edgecolor (str): Color of hexagon borders
        linewidth (float): Width of hexagon borders

    Returns:
        Tuple[list[RegularPolygon], float, float, float, float]: (patches, x_min, x_max, y_min, y_max)
    """
    # Convert to numpy if needed
    if isinstance(map_data, torch.Tensor):
        map_data = map_data.detach().cpu().numpy()

    rows, cols = map_data.shape
    patches = []

    # Get colormap
    cmap = cmap or plt.cm.get_cmap("viridis")

    # Normalize data for colormap (handle NaN values)
    valid_mask = ~np.isnan(map_data)
    if norm is None:
        if valid_mask.any():
            vmin = np.nanmin(map_data)
            vmax = np.nanmax(map_data)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=0, vmax=1)

    # Create hexagon for each grid cell
    for row in range(rows):
        for col in range(cols):
            # Get hexagon center coordinates
            center_x, center_y = grid_to_hex_coords(row, col)

            # Get color for this cell and use white for NaN values
            value = map_data[row, col]
            facecolor = "white" if np.isnan(value) else cmap(norm(value))

            # Create hexagon patch
            hexagon = create_hexagon_patch(
                center_x,
                center_y,
                radius=hex_radius,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
            patches.append(hexagon)

    # Calculate bounding box
    x_min, x_max, y_min, y_max = calculate_hex_dimensions(rows, cols, hex_radius)

    return patches, x_min, x_max, y_min, y_max
