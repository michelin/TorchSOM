"""Utility functions for grid operations."""

import math

import torch

# NOTE: Coordinate conversion functions moved to hexagonal_coordinates.py to eliminate duplication and provide single source of truth.


def create_mesh_grid(
    x: int,
    y: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a mesh grid for neighborhood calculations.

    The function returns two 2D tensors representing the x-coordinates and y-coordinates
    of a grid of shape (x, y). This is useful for computing distance-based neighborhood functions
    in Self-Organizing Maps (SOM).

    Args:
        x (int): Number of rows (height of the grid).
        y (int): Number of columns (width of the grid).
        device (str): The device on which tensors should be allocated ('cpu' or 'cuda').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two tensors (xx, yy) of shape (x, y), representing the x and y coordinates of the mesh grid.
    """
    x_tensor, y_tensor = torch.arange(x, device=device), torch.arange(
        y, device=device
    )  # Shape: (x) and (y)
    x_meshgrid, y_meshgrid = torch.meshgrid(
        x_tensor, y_tensor, indexing="ij"
    )  # Create 2D meshgrid of shapes (x, y): xx contains x-coordinates, yy contains y-coordinates
    return (
        x_meshgrid.float(),
        y_meshgrid.float(),
    )  # convert a torch(int) into a torch(float)


def adjust_meshgrid_topology(
    xx: torch.Tensor,
    yy: torch.Tensor,
    topology: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjust coordinates based on topology.

    Args:
        xx (torch.Tensor): Mesh grid of x coordinates
        yy (torch.Tensor): Mesh grid of y coordinates
        topology (str): SOM configuration, usually rectangular or hexagonal

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Adjusted x and y mesh grids for a hexagonal topology.
    """
    if topology == "hexagonal":
        # Create new tensors to avoid modifying in-place
        adjusted_xx = xx.clone()
        adjusted_yy = yy.clone()

        """
        Use even-r offset coordinate system (consistent with visualization)
            1. Even rows (0, 2, 4...): no horizontal shift
            2. Odd rows (1, 3, 5...): shift right by 0.5
        """
        adjusted_xx[1::2] += 0.5
        adjusted_yy *= math.sqrt(3) / 2

        return adjusted_xx, adjusted_yy

    return xx, yy


# def convert_to_axial_coords(
#     row: int,
#     col: int,
# ) -> tuple[float, float]:
#     """Convert grid coordinates to axial coordinates for hexagonal grid.

#     Uses even-r layout where even rows are shifted left by 0.5.
#     This matches the layout used in adjust_meshgrid_topology.

#     Args:
#         row (int): Grid row coordinate
#         col (int): Grid column coordinate

#     Returns:
#         tuple[float, float]: Axial coordinates (q, r)
#     """
#     q = col - 0.5 - row // 2 if row % 2 == 0 else col - row // 2
#     r = row
#     return q, r


# def offset_to_axial_coords(
#     row: int,
#     col: int,
# ) -> tuple[float, float]:  # pragma: no cover
#     """Convert offset coordinates to axial coordinates for hexagonal grid.

#     Alternative implementation that directly matches the mesh grid adjustment.

#     Args:
#         row (int): Grid row coordinate
#         col (int): Grid column coordinate

#     Returns:
#         tuple[float, float]: Axial coordinates (q, r)
#     """
#     # Direct conversion matching adjust_meshgrid_topology
#     q = col - (row - (row & 1)) / 2
#     r = row
#     return q, r


# def axial_distance(
#     q1: float,
#     r1: float,
#     q2: float,
#     r2: float,
# ) -> int:
#     """Calculate the distance between two hexes in axial coordinates.

#     Args:
#         q1 (float): column of first hex
#         r1 (float): row of first hex
#         q2 (float): column of second hex
#         r2 (float): row of second hex

#     Returns:
#         int: Distance in hex steps
#     """
#     # Convert axial to cube coordinates
#     x1, y1, z1 = q1, r1, -q1 - r1
#     x2, y2, z2 = q2, r2, -q2 - r2

#     # Manhattan distance divided by 2
#     return int((abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) / 2)
