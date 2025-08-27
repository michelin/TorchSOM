"""Abstract base class for all SOM variants."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseSOM(nn.Module, ABC):
    """Abstract base class for all SOM variants."""

    @abstractmethod
    def fit(
        self,
        data: torch.Tensor,
    ) -> tuple[list[float], list[float]]:
        """Train the SOM on the given data.

        Args:
            data (torch.Tensor): Input data tensor [batch_size, num_features]

        Returns:
            Tuple[List[float], List[float]]: Quantization and topographic errors [epoch]
        """
        pass

    @abstractmethod
    def identify_bmus(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Find best matching units for input data.

        Args:
            data (torch.Tensor): Input data tensor [batch_size, num_features] or [num_features]

        Returns:
            torch.Tensor: For single sample: Tensor of shape [2] with [row, col].
                          For batch: Tensor of shape [batch_size, 2] with [row, col] pairs
        """
        pass

    @abstractmethod
    def quantization_error(
        self,
        data: torch.Tensor,
    ) -> float:
        """Calculate quantization error.

        Args:
            data (torch.Tensor): Input data tensor [batch_size, num_features] or [num_features]

        Returns:
            float: Average quantization error value
        """
        pass

    @abstractmethod
    def topographic_error(
        self,
        data: torch.Tensor,
    ) -> float:
        """Calculate topographic error.

        Args:
            data (torch.Tensor): Input data tensor [batch_size, num_features] or [num_features]

        Returns:
            float: Topographic error ratio
        """
        pass

    @abstractmethod
    def initialize_weights(
        self,
        data: torch.Tensor,
        mode: Optional[str] = None,
    ) -> None:
        """Initialize the SOM weights.

        Args:
            data (torch.Tensor): Input data tensor [batch_size, num_features]
            mode (str, optional): Weight initialization method. Defaults to None.
        """
        pass
