from abc import ABC, abstractmethod
from typing import Union, Callable, Optional, Iterable

import ConfigSpace as CS
from ConfigSpace import hyperparameters as CSH

from pyPDP.utils.utils import ConfigSpaceHolder


class BlackboxFunction(ConfigSpaceHolder, ABC):
    def __init__(self, config_space: CS.ConfigurationSpace):
        super().__init__(config_space, seed=True)
        self.ndim = len(self.config_space.get_hyperparameters())
        self.__name__ = str(self)

    def __call__(self, **kwargs) -> float:
        config = CS.Configuration(self.config_space, values=kwargs)
        return self.value_from_config(config)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.ndim})"

    @abstractmethod
    def value_from_config(self, config: CS.Configuration) -> float:
        pass

    def pd_integral(self, *hyperparameters: Union[str, CSH.Hyperparameter], seed=None) -> 'BlackboxFunction':
        raise NotImplementedError(f"Integral not implemented for {self.__class__.__name__}")


class BlackboxFunctionND(BlackboxFunction, ABC):
    _default_lower = -5
    _default_upper = 5
    log = False

    def __init__(self, config_space: Optional[CS.ConfigurationSpace] = None):
        if config_space is None:
            config_space = config_space_nd(1, lower=self._default_lower, upper=self._default_upper, log=self.log)
        super().__init__(config_space)

    @classmethod
    def for_n_dimensions(cls, dimensions: int, *, lower=None, upper=None, seed=None):
        if lower is None:
            lower = cls._default_lower
        if upper is None:
            upper = cls._default_upper
        cs = config_space_nd(dimensions, lower=lower, upper=upper, log=cls.log, seed=seed)
        return cls(cs)

    # def pd_integral(self, *hyperparameters: Union[str, CSH.Hyperparameter], seed=None) -> 'BlackboxFunction':
    #     hps = convert_hyperparameters(hyperparameters)
    #     n_steps = 5000
    #     integral_numeric = 0
    #     meshgrid = np.meshgrid([
    #         np.linspace(hp.lower, hp.upper, 200)
    #         for hp in hps
    #         if isinstance(hp, CSH.NumericalHyperparameter)
    #     ])
    #     step_size = np.prod([val[1] for val in meshgrid])
    #     for values in zip(meshgrid):
    #         integral_numeric += self(x1=hp.lower + (i + 0.5) * step_size)
    #
    #     integral_numeric *= step_size


class CallableBlackboxFunction(BlackboxFunction):
    def __init__(self, function: Callable[[CS.Configuration], float], config_space: CS.ConfigurationSpace, name=None):
        super().__init__(config_space)
        self.f = function
        if name is not None:
            self.__name__ = name

    def value_from_config(self, config: CS.Configuration) -> float:
        return self.f(config)


def config_space_nd(
        dimensions: int, *,
        lower: Union[float, Iterable[float]] = -5,
        upper: Union[float, Iterable[float]] = 5,
        log: Union[bool, Iterable[bool]] = False,
        variable_prefix="x",
        seed=None
) -> CS.ConfigurationSpace:
    """
    Creates and returns a config space with `dimensions` dimensions.
    Parameters are named xi with i = [1,...,dimensions+1] (e.g. [x1, x2, x3] for `dimensions`=3)
    All Parameters are bounded between `lower` and `upper`.
    If `log` is True, parameters are specified as logarithmic

    If `lower`, `upper` or `log` are iterable, their values will be used for each respective axes
    """
    # Parse parameters
    assert int(dimensions) == dimensions
    dimensions = int(dimensions)  # Convert to integer
    if isinstance(lower, (float, int)):
        lower = [lower] * dimensions
    if isinstance(upper, (float, int)):
        upper = [upper] * dimensions
    if isinstance(log, bool):
        log = [log] * dimensions

    # Create Configspace
    cs = CS.ConfigurationSpace(seed=seed)
    for i, (low, high, is_log) in enumerate(zip(lower, upper, log)):
        name = f"{variable_prefix}{i + 1}"
        if low == high:
            x = CSH.Constant(name, value=low)
        else:
            x = CSH.UniformFloatHyperparameter(name, lower=low, upper=high, log=is_log)
        cs.add_hyperparameter(x)
    return cs
