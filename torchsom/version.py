"""Version of the torchsom package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torchsom")  # reads installed package version
except PackageNotFoundError:
    __version__ = "0.0.0"

# __version__ = "0.1.0"
