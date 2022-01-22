from abc import ABC, abstractmethod
from functools import cached_property
from typing import Tuple, Optional, List, Set

import numpy as np

from src.algorithms import Algorithm
from src.algorithms.ice import ICE
from src.surrogate_models import SurrogateModel
from src.utils.typing import SelectedHyperparameterType
from src.utils.utils import get_selected_idx

Sample = Tuple[np.ndarray, np.ndarray]  # configurations, variances


class Region:
    def __init__(self, x_points: np.ndarray, y_points: np.ndarray, y_variances: np.ndarray):
        """
        :param x_points: Shape: (num_points_in_region, num_grid_points, num_features)
        :param y_points: Shape: (num_points_in_region, num_grid_points)
        :param y_variances: Shape: (num_points_in_region, num_grid_points)
        """
        self.x_points = x_points
        self.y_points = y_points
        self.y_variances = y_variances

        assert len(self.x_points) == len(self.y_points) == len(self.y_variances)

    def __len__(self):
        return len(self.x_points)

    @cached_property
    def mean_confidence(self) -> float:
        return np.mean(self.y_variances).item()

    @cached_property
    def loss(self) -> float:
        # l2 loss calculation according to paper
        mean_variances = np.mean(self.y_variances, axis=0)
        pointwise_l2_loss = (self.y_variances - mean_variances) ** 2
        loss_sum = np.sum(pointwise_l2_loss, axis=None)

        return loss_sum.item()


class Partitioner(Algorithm, ABC):
    def __init__(self, surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 num_grid_points: int = 20,
                 num_samples: int = 1000):
        super().__init__(surrogate_model, selected_hyperparameter)
        self.num_grid_points = num_grid_points
        self.num_samples = num_samples

        # Properties
        self._ice: Optional[ICE] = None

        # save inputs of last calculation to save time
        self.max_depth = Optional[None]

        # get indices of selected hyperparameters
        cs = self.surrogate_model.config_space
        self.selected_idx: List[int] = get_selected_idx(self.selected_hyperparameter, cs)
        self.possible_split_param_idx: List[int] = list(set(range(self.num_features)) - set(self.selected_idx))

    @property
    def ice(self) -> ICE:
        if self._ice is None:
            self._ice = ICE(self.surrogate_model,
                            self.selected_hyperparameter,
                            self.num_samples,
                            self.num_grid_points_per_axis)
        return self._ice

    @abstractmethod
    def partition(self, max_depth: int = 1) -> List[Region]:
        pass


