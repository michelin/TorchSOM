"""Hexagonal coordinate system utilities.

This module provides coordinate conversion and distance calculation functions
for hexagonal grids using the even-r offset coordinate system.

Coordinate Systems:
- Offset coordinates: Traditional (row, col) grid indices
- Axial coordinates: Hexagonal (q, r) coordinate system
- Cube coordinates: 3D (x, y, z) representation where x + y + z = 0

References:
- https://www.redblobgames.com/grids/hexagons/
"""

import math


def offset_to_axial_coords(
    row: int,
    col: int,
) -> tuple[float, float]:
    """Convert offset coordinates to axial coordinates for hexagonal grid.

    Uses even-r offset coordinate system:
    - Even rows (0, 2, 4...): no horizontal shift
    - Odd rows (1, 3, 5...): shifted right by 0.5

    Args:
        row (int): Grid row coordinate (offset system)
        col (int): Grid column coordinate (offset system)

    Returns:
        tuple[float, float]: Axial coordinates (q, r)
    """
    q = col - (row - (row & 1)) / 2
    r = row
    return q, r


def axial_to_offset_coords(
    q: float,
    r: float,
) -> tuple[int, int]:
    """Convert axial coordinates to offset coordinates (even-r).

    Args:
        q (float): Axial q coordinate
        r (float): Axial r coordinate

    Returns:
        tuple[int, int]: (row, col) in offset coordinates
    """
    row = int(r)
    col = int(q + (r - (r & 1)) // 2)
    return row, col


def axial_to_cube_coords(
    q: float,
    r: float,
) -> tuple[float, float, float]:
    """Convert axial coordinates to cube coordinates.

    Args:
        q (float): Axial q coordinate
        r (float): Axial r coordinate

    Returns:
        tuple[float, float, float]: Cube coordinates (x, y, z)
    """
    x = q
    z = r
    y = -x - z
    return x, y, z


def cube_to_axial_coords(
    x: float,
    z: float,
) -> tuple[float, float]:
    """Convert cube coordinates to axial coordinates.

    Args:
        x (float): Cube x coordinate
        z (float): Cube z coordinate

    Returns:
        tuple[float, float]: Axial coordinates (q, r)
    """
    q = x
    r = z
    return q, r


def hexagonal_distance_axial(
    q1: float,
    r1: float,
    q2: float,
    r2: float,
) -> int:
    """Calculate distance between two hexagons using axial coordinates.

    Args:
        q1, r1: First hexagon's axial coordinates
        q2, r2: Second hexagon's axial coordinates

    Returns:
        int: Distance in hex steps
    """
    # Convert to cube coordinates for easier calculation
    x1, y1, z1 = axial_to_cube_coords(q1, r1)
    x2, y2, z2 = axial_to_cube_coords(q2, r2)

    # Manhattan distance in cube coordinates divided by 2
    return int((abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) / 2)


def hexagonal_distance_offset(
    row1: int,
    col1: int,
    row2: int,
    col2: int,
) -> int:
    """Calculate distance between two hexagons using offset coordinates.

    Args:
        row1, col1: First hexagon's offset coordinates
        row2, col2: Second hexagon's offset coordinates

    Returns:
        int: Distance in hex steps
    """
    # Convert to axial coordinates
    q1, r1 = offset_to_axial_coords(row1, col1)
    q2, r2 = offset_to_axial_coords(row2, col2)

    # Calculate distance in axial space
    return hexagonal_distance_axial(q1, r1, q2, r2)


def grid_to_display_coords(
    row: int,
    col: int,
    hex_radius: float = 1.0,
) -> tuple[float, float]:
    """Convert grid indices to display coordinates for visualization.

    Uses even-r offset coordinate system with proper hexagonal spacing.

    Args:
        row (int): Grid row index
        col (int): Grid column index
        hex_radius (float): Radius of hexagonal cells for scaling

    Returns:
        tuple[float, float]: (x, y) display coordinates for hexagon center
    """
    # Horizontal spacing - offset odd rows
    x = (col + (0.5 if row % 2 == 1 else 0.0)) * hex_radius * math.sqrt(3)

    # Vertical spacing - proper hexagonal ratio
    y = row * (hex_radius * 1.5)

    return x, y


def neighbors_offset(
    row: int,
    col: int,
) -> list[tuple[int, int]]:
    """Get the 6 neighboring coordinates in offset coordinate system.

    Args:
        row (int): Center hexagon row
        col (int): Center hexagon column

    Returns:
        list[tuple[int, int]]: List of (row, col) neighbor coordinates
    """
    if row % 2 == 0:  # Even row
        return [
            (row - 1, col - 1),
            (row - 1, col),  # Top neighbors
            (row, col - 1),
            (row, col + 1),  # Side neighbors
            (row + 1, col - 1),
            (row + 1, col),  # Bottom neighbors
        ]
    else:  # Odd row
        return [
            (row - 1, col),
            (row - 1, col + 1),  # Top neighbors
            (row, col - 1),
            (row, col + 1),  # Side neighbors
            (row + 1, col),
            (row + 1, col + 1),  # Bottom neighbors
        ]


def neighbors_axial(
    q: float,
    r: float,
) -> list[tuple[float, float]]:
    """Get the 6 neighboring coordinates in axial coordinate system.

    Args:
        q (float): Center hexagon q coordinate
        r (float): Center hexagon r coordinate

    Returns:
        list[tuple[float, float]]: List of (q, r) neighbor coordinates
    """
    return [
        (q + 1, r),
        (q + 1, r - 1),  # East, Northeast
        (q, r - 1),
        (q - 1, r),  # Northwest, West
        (q - 1, r + 1),
        (q, r + 1),  # Southwest, Southeast
    ]
