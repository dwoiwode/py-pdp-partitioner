from abc import ABC, abstractmethod
from typing import Iterable, Union

import ConfigSpace.hyperparameters as CSH

from src.surrogate_models import SurrogateModel


class Plottable(ABC):
    @abstractmethod
    def plot(self, *args, ax=None):
        pass


class Algorithm(ABC, Plottable):
    def __init__(self,
                 surrogate_model: SurrogateModel,
                 selected_hyperparameter: Union[CSH.Hyperparameter, Iterable[CSH.Hyperparameter]]):
        self.surrogate_model = surrogate_model

        if isinstance(selected_hyperparameter, CSH.Hyperparameter):
            selected_hyperparameter = (selected_hyperparameter,)
        self.selected_hyperparameter = tuple(selected_hyperparameter)
