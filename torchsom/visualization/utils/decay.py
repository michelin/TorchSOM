"""Utility functions for decay functions."""


def _lr_inverse_decay_to_zero(
    learning_rate: float,
    t: int,
    max_iter: int,
) -> float:
    """Decay function of learning rate that asymptotically approaches zero.

    Args:
        learning_rate (float): starting learning rate
        t (int): current iteration
        max_iter (int): maximum number of iterations

    Returns:
        float: learning rate at iteration t
    """
    gamma = max_iter / 100.0
    return learning_rate * gamma / (gamma + t)


def _lr_linear_decay_to_zero(
    learning_rate: float,
    t: int,
    max_iter: int,
) -> float:
    """Decay function of learning rate that linearly decreases to zero.

    Args:
        learning_rate (float): starting learning rate
        t (int): current iteration
        max_iter (int): maximum number of iterations

    Returns:
        float: learning rate at iteration t
    """
    return learning_rate * (1 - t / max_iter)


def _sig_inverse_decay_to_one(
    sigma: float,
    t: int,
    max_iter: int,
) -> float:
    """Decay function of sigma that asymptotically approaches one.

    Args:
        sigma (float): starting sigma
        t (int): current iteration
        max_iter (int): maximum number of iterations

    Returns:
        float: sigma at iteration t
    """
    gamma = (sigma - 1) / max_iter
    return sigma / (1 + (t * gamma))


def _sig_linear_decay_to_one(
    sigma: float,
    t: int,
    max_iter: int,
) -> float:
    """Decay function of sigma that linearly decreases to one.

    Args:
        sigma (float): starting sigma
        t (int): current iteration
        max_iter (int): maximum number of iterations

    Returns:
        float: sigma at iteration t
    """
    return sigma + (t * (1 - sigma) / max_iter)


def _asymptotic_decay(
    dynamic_parameter: float,
    t: int,
    max_iter: int,
) -> float:
    """Decay function of dynamic parameter that asymptotically approaches zero.

    Args:
        dynamic_parameter (float): initial value of the dynamic parameter
        t (int): current iteration
        max_iter (int): maximum number of iterations

    Returns:
        float: dynamic parameter at iteration t
    """
    return dynamic_parameter / (1 + t / (max_iter / 2))


DECAY_FUNCTIONS = {
    "lr_inverse_decay_to_zero": _lr_inverse_decay_to_zero,
    "sig_inverse_decay_to_one": _sig_inverse_decay_to_one,
    "lr_linear_decay_to_zero": _lr_linear_decay_to_zero,
    "sig_linear_decay_to_one": _sig_linear_decay_to_one,
    "asymptotic_decay": _asymptotic_decay,
}
