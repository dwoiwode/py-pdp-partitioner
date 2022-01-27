from abc import ABC, abstractmethod
from typing import Union, List, Callable, Any, Optional

import ConfigSpace as CS
from ConfigSpace import hyperparameters as CSH


class BlackboxFunction(ABC):
    def __init__(self, config_space: CS.ConfigurationSpace):
        self.config_space = config_space
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


class CallableBlackboxFunction(BlackboxFunction):
    def __init__(self, function: Callable[[CS.Configuration], float], config_space: CS.ConfigurationSpace):
        super(CallableBlackboxFunction, self).__init__(config_space)
        self.f = function

    def value_from_config(self, config: CS.Configuration) -> float:
        return self.f(config)


def config_space_nd(dimensions: int, *,
                    lower: float = -5,
                    upper: float = 5,
                    log: bool = False,
                    seed=None) -> CS.ConfigurationSpace:
    """
    Creates and returns a config space with `dimensions` dimensions.
    Parameters are named xi with i = [1,...,dimensions+1] (e.g. [x1, x2, x3] for `dimensions`=3)
    All Parameters are bounded between `lower` and `upper`.
    If `log` is True, parameters are specified as logarithmic
    """
    cs = CS.ConfigurationSpace(seed=seed)
    for i in range(dimensions):
        x = CSH.UniformFloatHyperparameter(f"x{i + 1}", lower=lower, upper=upper, log=log)
        cs.add_hyperparameter(x)
    return cs
