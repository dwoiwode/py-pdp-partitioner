"""
Collection of blackbox functions that can be minimized
"""
from typing import Union, List, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from pyPDP.blackbox_functions import BlackboxFunction, config_space_nd, CallableBlackboxFunction, BlackboxFunctionND
from pyPDP.utils.utils import convert_hyperparameters


class Square(BlackboxFunctionND):
    def value_from_config(self, config: CS.Configuration) -> float:
        return np.sum(np.square(list(config.values()))).item()


class NegativeSquare(BlackboxFunctionND):
    def value_from_config(self, config: CS.Configuration) -> float:
        return 1 - np.sum(np.square(list(config.values()))).item()


class Levy(BlackboxFunctionND):
    """
    https://www.sfu.ca/~ssurjano/levy.html.

    Input Domain:
    The function is usually evaluated on the hypercube x ∈ [-10, 10], for all x

    Global minimum:
    y = 0.0
    at *x = (1,...,1)
    """
    _default_lower = -10
    _default_upper = 10

    def value_from_config(self, config: CS.Configuration) -> float:
        x = np.asarray([config[f"x{i + 1}"] for i in range(self.ndim)])

        w = 1 + (x - 1) / 4
        term1 = np.power(np.sin(np.pi * w[0]), 2)

        term2 = np.square(w[:-1] - 1) * (1 + 10 * np.power(np.sin(np.pi * w[:-1] + 1), 2))
        term2 = np.sum(term2)

        term3 = np.power(w[-1] - 1, 2) * (1 + np.power(np.sin(2 * np.pi * w[-1]), 2))

        return term1 + term2 + term3


class Ackley(BlackboxFunctionND):
    """
    https://www.sfu.ca/~ssurjano/ackley.html

    Input Domain:
    The function is usually evaluated on the hypercube x ∈ [-32.768, 32.768], for all x although it may also be
    restricted to a smaller domain.

    Global minimum:
    y = 0.0
    at *x = (0,...,0)
    """
    _default_lower = -32.768
    _default_upper = 32.768

    a = 20
    b = 0.2
    c = 2 * np.pi

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


class StyblinskiTang(BlackboxFunctionND):
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

    def value_from_config(self, config: CS.Configuration) -> float:
        x = np.asarray([config[hp.name] for hp in self.config_space.get_hyperparameters()])

        return np.sum(np.power(x, 4) - 16 * np.power(x, 2) + 5 * x) / 2

    @staticmethod
    def _styblinski_tang_integral(x: float) -> float:
        return 0.5 * (0.2 * np.power(x, 5) - 16 / 3 * np.power(x, 3) + 2.5 * np.power(x, 2))

    def pd_integral(self, *hyperparameters: Union[str, CSH.Hyperparameter], seed=None,
                    return_offset: bool = False) -> Union[
        CallableBlackboxFunction, tuple[CallableBlackboxFunction, float]]:
        if len(hyperparameters) == 0:
            raise ValueError("Requires at least one hyperparameter for pd_integral")

        hyperparameters = convert_hyperparameters(hyperparameters, self.config_space)

        integral_offset = 0
        for hp in hyperparameters:
            # hp = hyperparameters[0]
            assert isinstance(hp, CSH.NumericalHyperparameter)
            lower = hp.lower
            upper = hp.upper
            integral_value = self._styblinski_tang_integral(upper) - self._styblinski_tang_integral(lower)
            integral_offset += integral_value / (upper - lower)

        hps = self.config_space.get_hyperparameters()
        reduced_cs = CS.ConfigurationSpace(seed=seed)
        hyperparameter_names = {hp.name for hp in hyperparameters}
        for hp in hps:
            if hp.name not in hyperparameter_names:
                reduced_cs.add_hyperparameter(hp)

        reduced_f = StyblinskiTang(reduced_cs)

        def integral(config: CS.Configuration):
            return reduced_f.value_from_config(config) + integral_offset

        if not return_offset:
            return CallableBlackboxFunction(integral, reduced_cs, name=f"{self.__name__} d({hyperparameter_names})")
        else:
            return (CallableBlackboxFunction(integral, reduced_cs, name=f"{self.__name__} d({hyperparameter_names})"),
                    integral_offset)





