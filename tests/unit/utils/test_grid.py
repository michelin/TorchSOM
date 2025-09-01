"""Tests for hexagonal grid coordinate utilities and distances."""

import pytest

from torchsom.utils.hexagonal_coordinates import (
    axial_to_cube_coords,
    axial_to_offset_coords,
    cube_to_axial_coords,
    grid_to_display_coords,
    hexagonal_distance_axial,
    hexagonal_distance_offset,
    neighbors_axial,
    neighbors_offset,
    offset_to_axial_coords,
)

pytestmark = [
    pytest.mark.unit,
]


class TestHexagonalCoordinates:
    def test_offset_to_axial_coords_even_rows(self) -> None:
        """Test offset to axial coordinate conversion for even rows."""
        # Test even rows (0, 2, 4...)
        q, r = offset_to_axial_coords(0, 0)
        assert (q, r) == (0, 0)

        q, r = offset_to_axial_coords(0, 2)
        assert (q, r) == (2, 0)

        q, r = offset_to_axial_coords(2, 1)
        assert (q, r) == (0, 2)

    def test_offset_to_axial_coords_odd_rows(self) -> None:
        """Test offset to axial coordinate conversion for odd rows."""
        # Test odd rows (1, 3, 5...)
        # For odd rows: q = col - (row - 1) / 2
        q, r = offset_to_axial_coords(1, 0)
        assert (q, r) == (0.0, 1)  # 0 - (1-1)/2 = 0

        q, r = offset_to_axial_coords(1, 1)
        assert (q, r) == (1.0, 1)  # 1 - (1-1)/2 = 1

        q, r = offset_to_axial_coords(3, 2)
        assert (q, r) == (1.0, 3)  # 2 - (3-1)/2 = 1

    def test_axial_to_offset_coords_roundtrip(
        self,
        hexagonal_test_coordinates: list[tuple[int, int]],
    ) -> None:
        """Test round-trip conversion: offset -> axial -> offset."""
        for row, col in hexagonal_test_coordinates:
            q, r = offset_to_axial_coords(row, col)
            row_back, col_back = axial_to_offset_coords(q, r)
            assert (row_back, col_back) == (row, col)

    def test_axial_to_cube_coords(self) -> None:
        """Test axial to cube coordinate conversion."""
        # Test basic conversions
        x, y, z = axial_to_cube_coords(0, 0)
        assert (x, y, z) == (0, 0, 0)
        assert x + y + z == 0  # Cube coordinate invariant

        x, y, z = axial_to_cube_coords(1, 0)
        assert (x, y, z) == (1, -1, 0)
        assert x + y + z == 0

        x, y, z = axial_to_cube_coords(0, 1)
        assert (x, y, z) == (0, -1, 1)
        assert x + y + z == 0

    def test_cube_to_axial_coords(self) -> None:
        """Test cube to axial coordinate conversion."""
        q, r = cube_to_axial_coords(0, 0)
        assert (q, r) == (0, 0)

        q, r = cube_to_axial_coords(1, 0)
        assert (q, r) == (1, 0)

        q, r = cube_to_axial_coords(0, 1)
        assert (q, r) == (0, 1)

    def test_cube_axial_roundtrip(self) -> None:
        """Test round-trip conversion: axial -> cube -> axial."""
        test_axial_coords = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, -1)]

        for q_orig, r_orig in test_axial_coords:
            x, y, z = axial_to_cube_coords(q_orig, r_orig)
            q_back, r_back = cube_to_axial_coords(x, z)
            assert (q_back, r_back) == (q_orig, r_orig)

    def test_hexagonal_distance_axial_basic(self) -> None:
        """Test basic hexagonal distance calculations in axial coordinates."""
        # Distance to self should be 0
        dist = hexagonal_distance_axial(0, 0, 0, 0)
        assert dist == 0

        # Distance to adjacent neighbors should be 1
        dist = hexagonal_distance_axial(0, 0, 1, 0)
        assert dist == 1

        dist = hexagonal_distance_axial(0, 0, 0, 1)
        assert dist == 1

        dist = hexagonal_distance_axial(0, 0, -1, 1)
        assert dist == 1

    def test_hexagonal_distance_axial_symmetry(self) -> None:
        """Test that hexagonal distance is symmetric."""
        test_pairs = [
            ((0, 0), (2, 1)),
            ((1, 0), (-1, 2)),
            ((-1, 1), (1, -1)),
        ]

        for (q1, r1), (q2, r2) in test_pairs:
            dist1 = hexagonal_distance_axial(q1, r1, q2, r2)
            dist2 = hexagonal_distance_axial(q2, r2, q1, r1)
            assert dist1 == dist2

    def test_hexagonal_distance_offset(
        self,
        hexagonal_test_coordinates: list[tuple[int, int]],
    ) -> None:
        """Test hexagonal distance calculation in offset coordinates."""
        # Test distance calculations match between offset and axial
        for i, (row1, col1) in enumerate(hexagonal_test_coordinates[:6]):
            for j, (row2, col2) in enumerate(hexagonal_test_coordinates[:6]):
                if i != j:
                    # Calculate using offset coordinates
                    dist_offset = hexagonal_distance_offset(row1, col1, row2, col2)

                    # Calculate using axial coordinates for verification
                    q1, r1 = offset_to_axial_coords(row1, col1)
                    q2, r2 = offset_to_axial_coords(row2, col2)
                    dist_axial = hexagonal_distance_axial(q1, r1, q2, r2)

                    assert dist_offset == dist_axial

    def test_grid_to_display_coords(self) -> None:
        """Test grid to display coordinate conversion."""
        # Test even row (no offset)
        x, y = grid_to_display_coords(0, 0, hex_radius=1.0)
        assert x == 0.0
        assert y == 0.0

        x, y = grid_to_display_coords(0, 1, hex_radius=1.0)
        assert abs(x - 1.732050807568877) < 1e-10  # sqrt(3)
        assert y == 0.0

        # Test odd row (with offset)
        x, y = grid_to_display_coords(1, 0, hex_radius=1.0)
        assert abs(x - 0.8660254037844386) < 1e-10  # sqrt(3)/2
        assert y == 1.5

        # Test with different radius
        x, y = grid_to_display_coords(0, 1, hex_radius=2.0)
        assert abs(x - 3.4641016151377544) < 1e-10  # 2*sqrt(3)
        assert y == 0.0

    def test_neighbors_offset_even_row(self) -> None:
        """Test neighbor finding for even rows in offset coordinates."""
        neighbors = neighbors_offset(0, 1)  # Even row
        expected = [
            (-1, 0),
            (-1, 1),  # Top neighbors
            (0, 0),
            (0, 2),  # Side neighbors
            (1, 0),
            (1, 1),  # Bottom neighbors
        ]
        assert neighbors == expected
        assert len(neighbors) == 6

    def test_neighbors_offset_odd_row(self) -> None:
        """Test neighbor finding for odd rows in offset coordinates."""
        neighbors = neighbors_offset(1, 1)  # Odd row
        expected = [
            (0, 1),
            (0, 2),  # Top neighbors
            (1, 0),
            (1, 2),  # Side neighbors
            (2, 1),
            (2, 2),  # Bottom neighbors
        ]
        assert neighbors == expected
        assert len(neighbors) == 6

    def test_neighbors_axial(self) -> None:
        """Test neighbor finding in axial coordinates."""
        neighbors = neighbors_axial(0, 0)
        expected = [
            (1, 0),
            (1, -1),  # East, Northeast
            (0, -1),
            (-1, 0),  # Northwest, West
            (-1, 1),
            (0, 1),  # Southwest, Southeast
        ]
        assert neighbors == expected
        assert len(neighbors) == 6

    def test_neighbors_distance_consistency(self) -> None:
        """Test that all neighbors are at distance 1."""
        center_q, center_r = 0, 0
        neighbors = neighbors_axial(center_q, center_r)

        for neighbor_q, neighbor_r in neighbors:
            distance = hexagonal_distance_axial(
                center_q, center_r, neighbor_q, neighbor_r
            )
            assert distance == 1

    def test_coordinate_conversion_edge_cases(self) -> None:
        """Test coordinate conversion with edge cases."""
        # Test negative coordinates
        q, r = offset_to_axial_coords(-1, -1)
        row, col = axial_to_offset_coords(q, r)
        assert (row, col) == (-1, -1)

        # Test large coordinates
        q, r = offset_to_axial_coords(100, 50)
        row, col = axial_to_offset_coords(q, r)
        assert (row, col) == (100, 50)

    def test_hexagonal_distance_triangle_inequality(self) -> None:
        """Test that hexagonal distance satisfies triangle inequality."""
        # Test triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        coords = [(0, 0), (2, 1), (-1, 3), (1, -2)]

        for i, (q1, r1) in enumerate(coords):
            for j, (q2, r2) in enumerate(coords):
                for k, (q3, r3) in enumerate(coords):
                    if i != j != k:
                        d_ac = hexagonal_distance_axial(q1, r1, q3, r3)
                        d_ab = hexagonal_distance_axial(q1, r1, q2, r2)
                        d_bc = hexagonal_distance_axial(q2, r2, q3, r3)
                        assert d_ac <= d_ab + d_bc
