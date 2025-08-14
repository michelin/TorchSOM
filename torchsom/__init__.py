"""Torchsom package."""

from .core import SOM, BaseSOM
from .utils.decay import DECAY_FUNCTIONS
from .utils.distances import DISTANCE_FUNCTIONS
from .utils.neighborhood import NEIGHBORHOOD_FUNCTIONS
from .visualization import SOMVisualizer, VisualizationConfig

# from .version import __version__

# Define what should be imported when using 'from torchsom import *'
__all__ = [
    "SOM",
    "BaseSOM",
    "DISTANCE_FUNCTIONS",
    "DECAY_FUNCTIONS",
    "NEIGHBORHOOD_FUNCTIONS",
    "SOMVisualizer",
    "VisualizationConfig",
]
