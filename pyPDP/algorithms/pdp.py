from functools import cached_property
from typing import Iterable, Optional

import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt

from pyPDP.algorithms import Algorithm
from pyPDP.algorithms.ice import ICE, ICECurve
from pyPDP.surrogate_models import SurrogateModel
from pyPDP.utils.typing import SelectedHyperparameterType


class PDP(Algorithm):
    def __init__(self,
                 surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 samples: np.ndarray,
                 num_grid_points_per_axis: int = 20,
                 seed=None):
        super().__init__(
            surrogate_model=surrogate_model,
            selected_hyperparameter=selected_hyperparameter,
            samples=samples,
            num_grid_points_per_axis=num_grid_points_per_axis,
            seed=seed
        )
        self._ice = None

    @property
    def ice(self):
        if self._ice is None:
            self._ice = ICE(
                surrogate_model=self.surrogate_model,
                selected_hyperparameter=self.selected_hyperparameter,
                samples=self.samples,
                num_grid_points_per_axis=self.num_grid_points_per_axis
            )
        return self._ice

    @property
    def grid_points(self) -> np.ndarray:
        return self.ice.grid_points

    @classmethod
    def from_ICE(cls, ice: ICE, seed=None) -> "PDP":
        pdp = PDP(
            surrogate_model=ice.surrogate_model,
            selected_hyperparameter=ice.selected_hyperparameter,
            samples=ice.samples,
            num_grid_points_per_axis=ice.num_grid_points_per_axis,
            seed=seed
        )
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

    @cached_property
    def as_ice_curve(self) -> ICECurve:
        return ICECurve(
            full_config_space=self.config_space,
            selected_hyperparameter=self.selected_hyperparameter,
            x_ice=self.x_pdp,
            y_ice=self.y_pdp,
            y_variances=self.y_variances,
            name="PDP"
        )

    def plot_values(self,
                    color="red",
                    ax: Optional[plt.Axes] = None):
        return self.as_ice_curve.plot_values(color=color, ax=ax)

    def plot_confidences(self,
                         line_color="red",
                         gradient_color="lightsalmon",
                         confidence_max_sigma=1.5,
                         ax: Optional[plt.Axes] = None):
        return self.as_ice_curve.plot_confidences(
            line_color=line_color,
            gradient_color=gradient_color,
            confidence_max_sigma=confidence_max_sigma,
            ax=ax
        )
