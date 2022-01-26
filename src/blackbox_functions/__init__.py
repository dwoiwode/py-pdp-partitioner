from abc import ABC, abstractmethod
from typing import Union, List, Callable, Any

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

    def pd_integral(self, hyperparameters: Union[List[CSH.Hyperparameter], CSH.Hyperparameter]) -> Callable[
        [Any], float]:
        raise NotImplementedError(f"Integral not implemented for {self.__class__.__name__}")


def config_space_nd(dimensions: int, *, lower: float = -5, upper: float = 5, seed=None) -> CS.ConfigurationSpace:
    cs = CS.ConfigurationSpace(seed=seed)
    for i in range(dimensions):
        x = CSH.UniformFloatHyperparameter(f"x{i + 1}", lower=lower, upper=upper)
        cs.add_hyperparameter(x)
    return cs
