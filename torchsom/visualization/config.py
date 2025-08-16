"""Configuration settings for SOM visualizations."""

from dataclasses import dataclass, field


@dataclass
class VisualizationConfig:
    """Configuration settings for SOM visualizations."""

    figsize: tuple[int, int] = (12, 8)
    fontsize: dict[str, int] = field(
        default_factory=lambda: {
            "title": 16,
            "axis": 13,
            "legend": 11,
        }
    )
    fontweight: dict[str, str] = field(
        default_factory=lambda: {
            "title": "bold",
            "axis": "normal",  # normal bold
            "legend": "normal",
        }
    )
    cmap: str = "viridis"
    dpi: int = 300
    grid_alpha: float = 0.3
    colorbar_pad: float = 0.01
    save_format: str = "png"

    # Hexagonal visualization settings
    hex_radius: float = 0.5
    hex_border_color: str = "black"
    hex_border_width: float = 0.3
