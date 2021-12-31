"""
Collection of blackbox functions that can be minimized
"""
import numpy as np


def _levy(*x: float) -> float:
    """
    https://www.sfu.ca/~ssurjano/levy.html.

    Input Domain:
    The function is usually evaluated on the hypercube x ∈ [-10, 10], for all x

    Global minimum:
    y = 0.0
    at *x = (1,...,1)
    """
    x = np.asarray(x)

    w = 1 + (x - 1) / 4
    term1 = np.power(np.sin(np.pi * w[0]), 2)

    term2 = np.square(w[:-1] - 1) * (1 + 10 * np.power(np.sin(np.pi * w[:-1] + 1), 2))
    term2 = np.sum(term2)

    term3 = np.power(w[-1] - 1, 2) * (1 + np.power(np.sin(2 * np.pi * w[-1]), 2))

    return term1 + term2 + term3


def _ackley(*x: float) -> float:
    """
    https://www.sfu.ca/~ssurjano/ackley.html

    Input Domain:
    The function is usually evaluated on the hypercube x ∈ [-32.768, 32.768], for all x although it may also be
    restricted to a smaller domain.

    Global minimum:
    y = 0.0
    at *x = (0,...,0)
    """
    a = 20
    b = 0.2
    c = 2 * np.pi

    x = np.asarray(x)
    d = len(x)
    term1 = np.exp(-b * np.sqrt(np.sum(np.square(x)) / d))
    term2 = np.exp(np.sum(np.cos(c * x) / d))

    return -a * term1 - term2 + a + np.exp(1)


def cross_in_tray(x1: float, x2: float) -> float:
    """
    https://www.sfu.ca/~ssurjano/crossit.html

    Input Domain:
    The function is usually evaluated on the square x1, x2 ∈ [-10, 10].

    Global Minimum:
    y = -2.06261
    at (1.3491, 1.3491),(-1.3491, 1.3491),(1.3491, -1.3491),(-1.3491, -1.3491)
    """
    term1 = np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(100 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))) + 1
    return -0.0001 * term1 ** 0.1


# Shortcuts
# Levy

def levy1D(x: float) -> float:
    return _levy(x)


def levy2D(x1: float, x2: float) -> float:
    return _levy(x1, x2)


# Ackley

def ackley1D(x: float) -> float:
    return _ackley(x)


def ackley2D(x1: float, x2: float) -> float:
    return _ackley(x1, x2)
