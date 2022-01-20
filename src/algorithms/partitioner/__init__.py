from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Iterable

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from src.algorithms import Algorithm
from src.utils.plotting import Plottable
from src.surrogate_models import SurrogateModel
from src.utils.typing import SelectedHyperparameterType

Sample = Tuple[np.ndarray, np.ndarray]  # configurations, variances


class Region:
    def __init__(self, x_points: np.ndarray, y_points: np.ndarray, y_variances: np.ndarray):
        """
        :param x_points: Shape: (num_points_in_region, num_gridpoints, num_features)
        :param y_points: Shape: (num_points_in_region, num_gridpoints)
        :param y_variances: Shape: (num_points_in_region, num_gridpoints)
        """
        self.x_points = x_points
        self.y_points = y_points
        self.y_variances = y_variances

        assert len(self.x_points) == len(self.y_points) == len(self.y_variances)


class Partitioner(Algorithm, ABC):
    def __init__(self, surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 num_grid_points: int = 20,
                 num_samples: int = 1000):
        super().__init__(surrogate_model, selected_hyperparameter)
        self.num_grid_points = num_grid_points
        self.num_samples = num_samples

        # Properties
        self._regions: Optional[Region] = None

    @property
    def regions(self):
        return
