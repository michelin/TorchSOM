"""Utility functions for topology."""

from typing import Union


def get_rectangular_offsets(
    neighborhood_order: int = 1,
) -> list[tuple[int, int]]:
    """Get neighbor offset coordinates for rectangular topology at any order.

    Args:
        neighborhood_order (int, optional): Order of neighborhood ring. Defaults to 1.

    Returns:
        List[Tuple[int, int]]: Coordinate offsets for rectangular grid

    Notes:
        Order 1: 8 neighbors
        Order 2: 16 neighbors
        Order 3: 24 neighbors
        Order 3+: All positions at Chebyshev distance (order-1)
    """
    if neighborhood_order < 1:
        raise ValueError("Neighborhood order must be >= 1")

    # Generate all positions at chebyshev distance : where max(|dx|, |dy|) = order (chebyshev distance)
    chebyshev_distance = neighborhood_order
    offsets = []
    for dx in range(-chebyshev_distance, chebyshev_distance + 1):
        for dy in range(-chebyshev_distance, chebyshev_distance + 1):
            if max(abs(dx), abs(dy)) == chebyshev_distance:
                offsets.append((dx, dy))

    return offsets


def get_hexagonal_offsets(
    neighborhood_order: int = 1,
) -> dict[str, list[tuple[int, int]]]:
    """Get neighbor offset coordinates for hexagonal topology at any order.

    Order n has 6*n elements.

    Args:
        neighborhood_order (int, optional): Order of neighborhood ring. Defaults to 1.

    Returns:
        Dict[str, List[Tuple[int, int]]]: Offsets for even and odd rows
    """
    if neighborhood_order < 1:
        raise ValueError("Neighborhood order must be >= 1")

    # Generate neighbors in axial coordinates using mathematical approach
    def generate_axial_ring(distance: int) -> list[tuple[int, int]]:
        """Generate all hexagonal neighbors at a specific distance in axial coordinates."""
        if distance == 0:
            return [(0, 0)]

        neighbors = []
        # Use cube coordinates for easier calculation
        # Start at (distance, -distance, 0) and walk around the ring
        x, y, z = distance, -distance, 0

        # Six directions to walk around the hexagonal ring
        directions = [
            (0, 1, -1),  # NE
            (-1, 1, 0),  # N
            (-1, 0, 1),  # NW
            (0, -1, 1),  # SW
            (1, -1, 0),  # S
            (1, 0, -1),  # SE
        ]

        for direction in directions:
            dx, dy, dz = direction
            for _i in range(distance):
                # Convert cube back to axial (q, r)
                q = x
                r = y
                neighbors.append((q, r))

                # Move to next position along this edge
                x += dx
                y += dy
                z += dz

        return neighbors

    # Generate axial coordinates for this order
    axial_neighbors = generate_axial_ring(neighborhood_order)

    # Convert axial (q,r) to offset (x,y) coordinates for even and odd rows
    even_offsets = []
    odd_offsets = []
    for q, r in axial_neighbors:
        even_col = q + (r + (r & 1)) // 2
        even_row = r
        even_offsets.append((even_row, even_col))
        odd_col = q + (r - (r & 1)) // 2
        odd_row = r
        odd_offsets.append((odd_row, odd_col))

    return {
        "even": even_offsets,
        "odd": odd_offsets,
    }


def get_all_neighbors_up_to_order(
    topology: str,
    max_order: int,
) -> Union[list[tuple[int, int]], dict[str, list[tuple[int, int]]]]:
    """Get all neighbors from order 1 up to max_order.

    Args:
        topology (str): "rectangular" or "hexagonal"
        max_order (int): Maximum neighborhood order to include

    Returns:
        All neighbor offsets from order 1 to max_order combined
    """
    if topology == "rectangular":
        all_offsets = []
        for order in range(1, max_order + 1):
            all_offsets.extend(get_rectangular_offsets(order))
        return all_offsets

    elif topology == "hexagonal":
        all_even_offsets = []
        all_odd_offsets = []
        for order in range(1, max_order + 1):
            order_offsets = get_hexagonal_offsets(order)
            all_even_offsets.extend(order_offsets["even"])
            all_odd_offsets.extend(order_offsets["odd"])
        return {
            "even": all_even_offsets,
            "odd": all_odd_offsets,
        }

    else:
        raise ValueError(f"Unsupported topology: {topology}")
