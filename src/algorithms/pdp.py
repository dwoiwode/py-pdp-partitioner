from functools import cached_property
from typing import Iterable, Optional

import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt

from src.algorithms import Algorithm
from src.algorithms.ice import ICE, ICECurve
from src.surrogate_models import SurrogateModel
from src.utils.typing import SelectedHyperparameterType


class PDP(Algorithm):
    def __init__(self,
                 surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 num_samples: int = 1000,
                 num_grid_points_per_axis: int = 20):
        super().__init__(surrogate_model, selected_hyperparameter, num_samples, num_grid_points_per_axis)
        self._ice = None

    @property
    def ice(self):
        if self._ice is None:
            self._ice = ICE(self.surrogate_model,
                            self.selected_hyperparameter,
                            self.num_samples,
                            self.num_grid_points_per_axis)
        return self._ice

    @classmethod
    def from_ICE(cls, ice: ICE) -> "PDP":
        pdp = PDP(ice.surrogate_model, ice.selected_hyperparameter, ice.num_grid_points_per_axis, ice.num_samples)
        pdp._ice = ice  # Use existing ice to save calculation time
        return pdp

    @cached_property
    def x_pdp(self) -> np.ndarray:
        return np.mean(self.ice.x_ice, axis=0)

    @cached_property
    def y_pdp(self) -> np.ndarray:
        return np.mean(self.ice.y_ice, axis=0)

    @cached_property
    def y_variances(self) -> np.ndarray:
        return np.mean(self.ice.y_variances, axis=0)

    def plot(self,
             line_color="red",
             gradient_color="xkcd:light red",
             with_confidence=True,
             ax: Optional[plt.Axes] = None):
        pdp = ICECurve(self.config_space, self.selected_hyperparameter,
                       self.x_pdp, self.y_pdp, self.y_variances,
                       name="PDP")
        pdp.plot(line_color=line_color, gradient_color=gradient_color,
                 with_confidence=with_confidence, ax=ax)
