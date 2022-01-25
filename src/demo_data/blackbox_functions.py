"""
Collection of blackbox functions that can be minimized
"""
import math

import numpy as np


def square(x1: float) -> float:
    return x1 ** 2


def neg_square(x1: float) -> float:
    return 1 - x1 ** 2


def square_2D(x1: float, x2: float) -> float:
    return x1 ** 2 + x2 ** 2


def levy(*x: float) -> float:
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


def ackley(*x: float) -> float:
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


def styblinski_tang(*x: float) -> float:
    """
    https://www.sfu.ca/~ssurjano/stybtang.html
    Example from the original paper

    Input Domain:
    The function is usually evaluated on the hypercube x ∈ [-5, 5] for all x.

    Global Minimum:
    d = number of dimensions
    y = -39.16599 * d
    at (-2.903534, ..., -2.903534)
    """
    x = np.asarray(x)

    return np.sum(np.power(x, 4) - 16 * np.power(x, 2) + 5 * x) / 2

# Integral helper functions
def styblinski_tang_integral(x1: float) -> float:
    return 0.5 * (0.2 * np.power(x1, 5) - 16/3 * np.power(x1, 3) + 2.5 * np.power(x1, 2))



# Shortcuts
def levy_1D(x1: float) -> float:
    return levy(x1)


def levy_2D(x1: float, x2: float) -> float:
    return levy(x1, x2)


def ackley_1D(x1: float) -> float:
    return ackley(x1)


def ackley_2D(x1: float, x2: float) -> float:
    return ackley(x1, x2)


def styblinski_tang_2D(x1: float, x2: float) -> float:
    return styblinski_tang(x1, x2)

def styblinski_tang_3D(x1: float, x2: float, x3: float) -> float:
    return styblinski_tang(x1, x2, x3)

def styblinski_tang_5D(x1: float, x2: float, x3: float, x4: float, x5: float) -> float:
    return styblinski_tang(x1, x2, x3, x4, x5)

def styblinski_tang_8D(x1: float, x2: float, x3: float, x4: float, x5: float, x6: float, x7: float, x8: float) -> float:
    return styblinski_tang(x1, x2, x3, x4, x5, x6, x7, x8)

def styblinski_tang_3D_int_2D(x1: float, x2: float, lower: float = -5, upper: float = 5) -> float:
    lower_term = styblinski_tang_2D(x1, x2) * lower + styblinski_tang_integral(lower)
    upper_term = styblinski_tang_2D(x1, x2) * upper + styblinski_tang_integral(upper)
    return (upper_term - lower_term) / (upper - lower)  # normalization

def styblinski_tang_3D_int_1D(x1: float, lower_x2: float = -5, upper_x2: float = 5, lower_x3: float = -5,
                              upper_x3: float = 5) -> float:
    term_x1_lower_lower = styblinski_tang(x1) * lower_x2 * lower_x3
    term_x1_lower_upper = styblinski_tang(x1) * lower_x2 * upper_x3
    term_x1_upper_lower = styblinski_tang(x1) * upper_x2 * lower_x3
    term_x1_upper_upper = styblinski_tang(x1) * upper_x2 * upper_x3
    term_x1 = term_x1_upper_upper - term_x1_upper_lower - term_x1_lower_upper + term_x1_lower_lower

    term_x2_lower_lower = styblinski_tang_integral(lower_x2) * lower_x3
    term_x2_lower_upper = styblinski_tang_integral(lower_x2) * upper_x3
    term_x2_upper_lower = styblinski_tang_integral(upper_x2) * lower_x3
    term_x2_upper_upper = styblinski_tang_integral(upper_x2) * upper_x3
    term_x2 = term_x2_upper_upper - term_x2_upper_lower - term_x2_lower_upper + term_x2_lower_lower

    term_x3_lower_lower = styblinski_tang_integral(lower_x3) * lower_x2
    term_x3_lower_upper = styblinski_tang_integral(lower_x3) * upper_x2
    term_x3_upper_lower = styblinski_tang_integral(upper_x3) * lower_x2
    term_x3_upper_upper = styblinski_tang_integral(upper_x3) * upper_x2
    term_x3 = term_x3_upper_upper - term_x3_upper_lower - term_x3_lower_upper + term_x3_lower_lower

    return (term_x1 + term_x2 + term_x3) / ((upper_x2 - lower_x2) * (upper_x3 - lower_x3))
