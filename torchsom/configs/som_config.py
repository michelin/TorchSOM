from typing import Literal, Optional, Union

import torch
from pydantic import BaseModel, Field, validator


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
        "cuda" if torch.cuda.is_available() else "cpu",
        description="Device for tensor computations",
    )
    random_seed: int = Field(42, description="Random seed for reproducibility")


# SOM:

#   x: 100 # Number of rows: 100
#   y: 75 # Number of columns: 75
#   topology: "rectangular" # Grid topology: "rectangular" or "hexagonal"
#   epochs: 50 # Number of epochs to train the model: 50
#   batch_size: 256 # Number of samples to train the model: 64
#   learning_rate: 0.95 # Initial learning rate: 0.95
#   sigma: 5.0 # Initial spread of the neighborhood function: 5.0
#   neighborhood_function: "gaussian" # Neighborhood function: "gaussian", "mexican_hat", "bubble", "triangle"
#   distance_function: "euclidean" # Function to calculate distance between data and neurons: "euclidean" "cosine" "manhattan" "chebyshev" "weighted_euclidean" (need to provide weights_proportion)
#   lr_decay_function: "asymptotic_decay" # Learning rate scheduler: "lr_inverse_decay_to_zero", "lr_linear_decay_to_zero", "asymptotic_decay"
#   sigma_decay_function: "asymptotic_decay" # Sigma scheduler: "sig_inverse_decay_to_one" "sig_linear_decay_to_one" "asymptotic_decay"
#   initialization_mode: "pca" # Weights initialization method: "random" or "pca"
#   neighborhood_order: 3 # Indicate which neighbors should be considered in SOM distance map and JITL buffer: 1
# !  device: "cuda" # Device for tensor computations: "cuda" or "cpu"
# !  random_seed: 42 # Random seed for reproducibility
