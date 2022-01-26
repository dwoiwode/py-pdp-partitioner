"""
Collection of blackbox functions that can be minimized
"""

import ConfigSpace as CS
import numpy as np

from src.blackbox_functions import BlackboxFunction, config_space_nd


class Square(BlackboxFunction):
    def __init__(self, dimensions: int, *, lower: float = -5, upper: float = 5, seed=None):
        super().__init__(config_space_nd(dimensions, lower=lower, upper=upper, seed=seed))

    def value_from_config(self, config: CS.Configuration) -> float:
        return np.sum(np.square(list(config.values()))).item()


class NegativeSquare(BlackboxFunction):
    def __init__(self, dimensions: int, *, lower: float = -5, upper: float = 5, seed=None):
        super().__init__(config_space_nd(dimensions, lower=lower, upper=upper, seed=seed))

    def value_from_config(self, config: CS.Configuration) -> float:
        return 1 - np.sum(np.square(list(config.values()))).item()


class Levy(BlackboxFunction):
    """
    https://www.sfu.ca/~ssurjano/levy.html.

    Input Domain:
    The function is usually evaluated on the hypercube x ∈ [-10, 10], for all x

    Global minimum:
    y = 0.0
    at *x = (1,...,1)
    """

    def __init__(self, dimensions: int, *, lower: float = -10, upper: float = 10, seed=None):
        super().__init__(config_space_nd(dimensions, lower=lower, upper=upper, seed=seed))

    def value_from_config(self, config: CS.Configuration) -> float:
        x = np.asarray([config[f"x{i + 1}"] for i in range(self.ndim)])

        w = 1 + (x - 1) / 4
        term1 = np.power(np.sin(np.pi * w[0]), 2)

        term2 = np.square(w[:-1] - 1) * (1 + 10 * np.power(np.sin(np.pi * w[:-1] + 1), 2))
        term2 = np.sum(term2)

        term3 = np.power(w[-1] - 1, 2) * (1 + np.power(np.sin(2 * np.pi * w[-1]), 2))

        return term1 + term2 + term3


class Ackley(BlackboxFunction):
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

    def __init__(self, dimensions: int, *, lower: float = -32.768, upper: float = 32.768, seed=None):
        super().__init__(config_space_nd(dimensions, lower=lower, upper=upper, seed=seed))

    def value_from_config(self, config: CS.Configuration) -> float:
        d = self.ndim
        x = np.asarray([config[f"x{i + 1}"] for i in range(d)])

        term1 = np.exp(-self.b * np.sqrt(np.sum(np.square(x)) / d))
        term2 = np.exp(np.sum(np.cos(self.c * x) / d))

        return -self.a * term1 - term2 + self.a + np.exp(1)


class CrossInTray(BlackboxFunction):
    """
    https://www.sfu.ca/~ssurjano/crossit.html

    Input Domain:
    The function is usually evaluated on the square x1, x2 ∈ [-10, 10].

    Global Minimum:
    y = -2.06261
    at (1.3491, 1.3491),(-1.3491, 1.3491),(1.3491, -1.3491),(-1.3491, -1.3491)
    """

    def __init__(self, *, lower: float = -10, upper: float = 10, seed=None):
        super().__init__(config_space_nd(2, lower=lower, upper=upper, seed=seed))

    def value_from_config(self, config: CS.Configuration) -> float:
        x1 = config["x1"]
        x2 = config["x2"]

        term1 = np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(100 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))) + 1
        return -0.0001 * term1 ** 0.1


class StyblinskiTang(BlackboxFunction):
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

    def __init__(self, dimensions: int, *, lower=-5, upper=5, seed=None):
        super().__init__(config_space_nd(dimensions, lower=lower, upper=upper, seed=seed))

    def value_from_config(self, config: CS.Configuration) -> float:
        x = np.asarray([config[f"x{i + 1}"] for i in range(self.ndim)])

        return np.sum(np.power(x, 4) - 16 * np.power(x, 2) + 5 * x) / 2


# Integral helper functions
def styblinski_tang_integral(x1: float) -> float:
    return 0.5 * (0.2 * np.power(x1, 5) - 16 / 3 * np.power(x1, 3) + 2.5 * np.power(x1, 2))

# Shortcuts
def styblinski_tang_3D_int_2D(x1: float, x2: float, lower: float = -5, upper: float = 5) -> float:
    styblinski_tang_2D = StyblinskiTang(2)
    lower_term = styblinski_tang_2D(x1=x1, x2=x2) * lower + styblinski_tang_integral(lower)
    upper_term = styblinski_tang_2D(x1=x1, x2=x2) * upper + styblinski_tang_integral(upper)
    return (upper_term - lower_term) / (upper - lower)  # normalization


def styblinski_tang_3D_int_1D(x1: float, lower_x2: float = -5, upper_x2: float = 5, lower_x3: float = -5,
                              upper_x3: float = 5) -> float:
    styblinski_tang = StyblinskiTang(1)
    term_x1_lower_lower = styblinski_tang(x1=x1) * lower_x2 * lower_x3
    term_x1_lower_upper = styblinski_tang(x1=x1) * lower_x2 * upper_x3
    term_x1_upper_lower = styblinski_tang(x1=x1) * upper_x2 * lower_x3
    term_x1_upper_upper = styblinski_tang(x1=x1) * upper_x2 * upper_x3
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
