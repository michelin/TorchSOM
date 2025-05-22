from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class VisualizationConfig:
    """Configuration settings for SOM visualizations."""

    figsize: Tuple[int, int] = (12, 8)
    fontsize: Dict[str, int] = field(
        default_factory=lambda: {
            "title": 16,
            "axis": 13,
            "legend": 11,
        }
    )
    fontweight: Dict[str, str] = field(
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
    hexgrid_size: Optional[int] = None
