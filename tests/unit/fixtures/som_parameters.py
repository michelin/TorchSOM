"""SOM config fixtures."""

import pytest


@pytest.fixture(
    params=[
        "rectangular",
        "hexagonal",
    ]
)
def topology(
    request: pytest.FixtureRequest,
) -> str:
    """Parametrized topology fixture for testing both topologies."""
    return request.param


@pytest.fixture(
    params=[
        "euclidean",
        "cosine",
        "manhattan",
        "chebyshev",
    ]
)
def distance_function(
    request: pytest.FixtureRequest,
) -> str:
    """Parametrized distance function fixture."""
    return request.param


@pytest.fixture(
    params=[
        "gaussian",
        "bubble",
        "triangle",
        "mexican_hat",
    ]
)
def neighborhood_function(
    request: pytest.FixtureRequest,
) -> str:
    """Parametrized neighborhood function fixture."""
    return request.param


@pytest.fixture(
    params=[
        "lr_inverse_decay_to_zero",
        "lr_linear_decay_to_zero",
        "asymptotic_decay",
    ]
)
def lr_decay_function(
    request: pytest.FixtureRequest,
) -> str:
    """Parametrized learning rate decay function fixture."""
    return request.param


@pytest.fixture(
    params=[
        "sig_inverse_decay_to_one",
        "sig_linear_decay_to_one",
        "asymptotic_decay",
    ]
)
def sigma_decay_function(
    request: pytest.FixtureRequest,
) -> str:
    """Parametrized sigma decay function fixture."""
    return request.param


@pytest.fixture(
    params=[
        "random",
        "pca",
    ]
)
def initialization_mode(
    request: pytest.FixtureRequest,
) -> str:
    """Parametrized initialization mode fixture."""
    return request.param


@pytest.fixture(
    params=[
        "mean",
        "std",
    ]
)
def reduction_parameter(
    request: pytest.FixtureRequest,
) -> str:
    """Parametrized reduction parameter fixture."""
    return request.param


@pytest.fixture(
    params=[
        True,
        False,
    ]
)
def return_indices(
    request: pytest.FixtureRequest,
) -> bool:
    """Parametrized return indices fixture."""
    return request.param


@pytest.fixture(
    params=[
        "kmeans",
        "gmm",
        # "hdbscan",
    ]
)
def clustering_method(
    request: pytest.FixtureRequest,
) -> str:
    """Parametrized clustering method fixture."""
    return request.param


@pytest.fixture(
    params=[
        "weights",
        "positions",
        "combined",
    ]
)
def clustering_space(
    request: pytest.FixtureRequest,
) -> str:
    """Parametrized clustering space fixture."""
    return request.param
