from abc import ABC, abstractmethod
from functools import cached_property
from typing import Tuple, Optional, List, Callable

import ConfigSpace as CS
import numpy as np
import ConfigSpace.hyperparameters as CSH

from src.algorithms import Algorithm
from src.algorithms.ice import ICE
from src.surrogate_models import SurrogateModel
from src.utils.typing import SelectedHyperparameterType
from src.utils.utils import get_selected_idx, unscale_float

from scipy.stats import norm

Sample = Tuple[np.ndarray, np.ndarray]  # configurations, variances


class Region:
    def __init__(self, x_points: np.ndarray, y_points: np.ndarray, y_variances: np.ndarray,
                 config_space: CS.ConfigurationSpace, selected_hyperparameter: SelectedHyperparameterType):
        """
        :param x_points: Shape: (num_points_in_region, num_grid_points, num_features)
        :param y_points: Shape: (num_points_in_region, num_grid_points)
        :param y_variances: Shape: (num_points_in_region, num_grid_points)
        """
        self.x_points = x_points
        self.y_points = y_points
        self.y_variances = y_variances
        self.config_space = config_space
        self.selected_hyperparameter = selected_hyperparameter

        assert len(self.x_points) == len(self.y_points) == len(self.y_variances)
        assert self.x_points.shape[1] == self.y_points.shape[1] == self.y_variances.shape[1]

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

    def negative_log_likelihood(self, true_pd_function: Callable[[float], float]) -> float:
        num_grid_points = self.x_points.shape[1]

        # true pd should have one or two inputs depending on dimensions chosen TODO: 2d
        hyperparameter_idx = self.config_space.get_idx_by_hyperparameter_name(list(self.selected_hyperparameter)[0].name)
        true_y = np.ndarray(shape=(num_grid_points,))
        for i in range(num_grid_points):
            unscaled_x = unscale_float(self.x_points[0, i, hyperparameter_idx], self.config_space,
                                       list(self.selected_hyperparameter)[0])
            true_y[i] = true_pd_function(unscaled_x)

        # regions pdp estimate:
        pdp_y_points = np.mean(self.y_points, axis=0)
        pdp_y_variances = np.mean(self.y_variances, axis=0)
        pdp_y_std = np.sqrt(pdp_y_variances)

        log_prob = norm.logpdf(true_y, loc=pdp_y_points, scale=pdp_y_std)
        result = - np.mean(log_prob)
        return result


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
        self.possible_split_parameters: List[CSH.Hyperparameter] = [
            hp for hp in cs.get_hyperparameters()
            if hp.name != selected_hyperparameter.name
        ]

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


