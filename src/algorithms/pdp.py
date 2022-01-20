from functools import cached_property

import ConfigSpace as CS
import numpy as np

from src.algorithms import Algorithm
from src.algorithms.ice import ICE
from src.surrogate_models import SurrogateModel


class PDP(Algorithm):
    def __init__(self,
                 surrogate_model: SurrogateModel,
                 cs: CS.ConfigurationSpace,
                 num_grid_points_per_axis: int = 20,
                 num_samples: int = 1000):
        super(PDP, self).__init__(surrogate_model, cs)
        self.ice = ICE(surrogate_model, cs, num_grid_points_per_axis=num_grid_points_per_axis, num_samples=num_samples)

    @classmethod
    def from_ICE(cls, ice: ICE) -> "PDP":
        pdp = PDP(ice.surrogate_model, ice.cs, ice.num_grid_points_per_axis, ice.num_samples)
        pdp.ice = ice
        return pdp

    @cached_property
    def x_pdp(self) -> np.ndarray:
        return np.mean(self.ice.x_ice, axis=0)

    @cached_property
    def y_pdp(self) -> np.ndarray:
        return np.mean(self.ice.y_ice, axis=0)

    @cached_property
    def y_variance(self) -> np.ndarray:
        return np.mean(self.ice.y_variances, axis=0)
