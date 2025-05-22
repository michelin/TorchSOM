from typing import Dict, List, Tuple


def get_hexagonal_offsets(
    neighborhood_order: int = 1,
) -> Dict[str, List[Tuple[int, int]]]:
    """Get neighbor offset coordinates for hexagonal topology.

    Args:
        neighborhood_order (int, optional): Order of neighborhood ring. Defaults to 1.

    Returns:
        Dict[str, List[Tuple[int, int]]]: Offsets for even and odd rows

    Notes:
        Neighboring ring of order 1 has 6 hexagonal elements,
        Neighboring ring of order 2 has 12 hexagonal elements,
        Neighboring ring of order 3 has 18 hexagonal elements
    """
    if neighborhood_order == 1:
        return {
            "even": [(0, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)],
            "odd": [(0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)],
        }
    elif neighborhood_order == 2:
        return {
            "even": [
                (0, 2),
                (1, 1),
                (2, 0),
                (2, -1),
                (2, -2),
                (1, -2),
                (0, -2),
                (-1, -2),
                (-2, -1),
                (-2, 0),
                (-1, 1),
                (-1, 2),
            ],
            "odd": [
                (0, 2),
                (1, 2),
                (2, 1),
                (2, 0),
                (1, -1),
                (0, -2),
                (-1, -1),
                (-2, -1),
                (-2, 0),
                (-2, 1),
                (-1, 2),
                (-1, 3),
            ],
        }
    elif neighborhood_order == 3:
        return {
            "even": [
                (0, 3),
                (1, 2),
                (2, 1),
                (3, 0),
                (3, -1),
                (3, -2),
                (3, -3),
                (2, -3),
                (1, -3),
                (0, -3),
                (-1, -3),
                (-2, -2),
                (-3, -1),
                (-3, 0),
                (-2, 1),
                (-1, 2),
                (-1, 3),
                (-2, 3),
            ],
            "odd": [
                (0, 3),
                (1, 3),
                (2, 2),
                (3, 1),
                (3, 0),
                (3, -1),
                (2, -2),
                (1, -2),
                (0, -3),
                (-1, -2),
                (-2, -2),
                (-3, -1),
                (-3, 0),
                (-3, 1),
                (-3, 2),
                (-2, 2),
                (-1, 3),
                (-1, 4),
            ],
        }
    else:
        raise ValueError(f"Neighborhood order {neighborhood_order} not supported")


def get_rectangular_offsets(
    neighborhood_order: int = 1,
) -> List[Tuple[int, int]]:
    """Get neighbor offset coordinates for rectangular topology.

    Args:
        neighborhood_order (int, optional): Order of neighborhood ring. Defaults to 1.

    Returns:
        List[Tuple[int, int]]: Coordinate offsets for rectangular grid

    Notes:
        Neighboring ring of order 1 has 4 elements: Von Neumann neighborhood (orthogonal only),
        Neighboring ring of order 2 has 4 elements: Diagonal neighbors,
        Neighboring ring of order 3 has 16 elements: outer edge of 5x5 grid (without inner squares)
    """
    if neighborhood_order == 1:
        return [(0, 1), (1, 0), (0, -1), (-1, 0)]
    elif neighborhood_order == 2:
        return [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    elif neighborhood_order == 3:
        return [
            (-2, -1),
            (-2, 0),
            (-2, 1),
            (2, -1),
            (2, 0),
            (2, 1),
            (-1, -2),
            (0, -2),
            (1, -2),
            (-1, 2),
            (0, 2),
            (1, 2),
            (-2, -2),
            (-2, 2),
            (2, -2),
            (2, 2),
        ]
    else:
        raise ValueError(f"Neighborhood order {neighborhood_order} not supported")
