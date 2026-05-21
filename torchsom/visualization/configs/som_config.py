"""Configuration for SOM parameters using pydantic for validation."""

from typing import Literal

import torch
from pydantic import BaseModel, Field


class SOMConfig(BaseModel):
    """Configuration for SOM parameters using pydantic for validation."""

    # Map structure parameters
    x: int = Field(..., description="Number of rows in the map", gt=0)
    y: int = Field(..., description="Number of columns in the map", gt=0)
    topology: Literal["rectangular", "hexagonal"] = Field(
        "rectangular", description="Grid topology"
    )

    # Training parameters
    epochs: int = Field(10, description="Number of training epochs", ge=1)
    batch_size: int = Field(5, description="Batch size for training", ge=1)
    learning_rate: float = Field(0.5, description="Initial learning rate", gt=0)
    sigma: float = Field(1.0, description="Initial neighborhood radius", gt=0)

    # Function choices
    neighborhood_function: Literal["gaussian", "mexican_hat", "bubble", "triangle"] = (
        Field(
            "gaussian",
            description="Function to determine neuron neighborhood influence",
        )
    )
    distance_function: Literal["euclidean", "cosine", "manhattan", "chebyshev"] = Field(
        "euclidean", description="Function to compute distances"
    )
    lr_decay_function: Literal[
        "lr_inverse_decay_to_zero", "lr_linear_decay_to_zero", "asymptotic_decay"
    ] = Field("asymptotic_decay", description="Learning rate decay function")
    sigma_decay_function: Literal[
        "sig_inverse_decay_to_one", "sig_linear_decay_to_one", "asymptotic_decay"
    ] = Field("asymptotic_decay", description="Sigma decay function")
    initialization_mode: Literal["random", "pca"] = Field(
        "random", description="Weight initialization method"
    )

    # Other parameters
    neighborhood_order: int = Field(
        1, description="Neighborhood order for distance calculations", ge=1
    )
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device for tensor computations",
    )
    random_seed: int = Field(42, description="Random seed for reproducibility")
