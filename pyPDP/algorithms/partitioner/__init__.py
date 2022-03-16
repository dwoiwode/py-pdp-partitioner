from abc import ABC, abstractmethod
from functools import cached_property
from typing import Tuple, Optional, List, Callable

import ConfigSpace as CS
import numpy as np
import ConfigSpace.hyperparameters as CSH
from matplotlib import pyplot as plt

from pyPDP.algorithms import Algorithm
from pyPDP.algorithms.ice import ICE, ICECurve
from pyPDP.blackbox_functions import BlackboxFunction
from pyPDP.surrogate_models import SurrogateModel
from pyPDP.utils.typing import SelectedHyperparameterType
from pyPDP.utils.utils import unscale_float, calculate_log_delta, ConfigSpaceHolder, get_hyperparameters

from scipy.stats import norm

Sample = Tuple[np.ndarray, np.ndarray]  # configurations, variances


class Region(ConfigSpaceHolder):
    def __init__(
            self,
            x_points: np.ndarray,
            y_points: np.ndarray,
            y_variances: np.ndarray,
            config_space: CS.ConfigurationSpace,
            selected_hyperparameter: SelectedHyperparameterType
    ):
        """
        :param x_points: Shape: (num_points_in_region, num_grid_points, num_features)
        :param y_points: Shape: (num_points_in_region, num_grid_points)
        :param y_variances: Shape: (num_points_in_region, num_grid_points)
        """
        super().__init__(config_space)
        self.x_points = x_points
        self.y_points = y_points
        self.y_variances = y_variances
        if isinstance(selected_hyperparameter, CSH.Hyperparameter):
            selected_hyperparameter = [selected_hyperparameter]
        self.selected_hyperparameter = tuple(selected_hyperparameter)

        assert len(self.x_points) == len(self.y_points) == len(self.y_variances)
        assert self.x_points.shape[1] == self.y_points.shape[1] == self.y_variances.shape[1]

    def __len__(self):
        return len(self.x_points)

    @cached_property
    def mean_confidence(self) -> float:
        return np.mean(np.sqrt(self.y_variances)).item()

    @cached_property
    def loss(self) -> float:
        # l2 loss calculation according to paper
        mean_variances = np.mean(self.y_variances, axis=0)
        pointwise_l2_loss = (self.y_variances - mean_variances) ** 2
        loss_sum = np.sum(pointwise_l2_loss, axis=None)
        return loss_sum.item()

    def negative_log_likelihood(self, true_function: BlackboxFunction) -> float:
        num_grid_points = self.x_points.shape[1]

        # true pd should have one or two inputs depending on dimensions chosen TODO: 2d
        hyperparameter_idx = self.config_space.get_idx_by_hyperparameter_name(self.selected_hyperparameter[0].name)
        true_y = np.ndarray(shape=(num_grid_points,))
        selected_hyperparameter_names = {hp.name for hp in self.selected_hyperparameter}
        not_selected_hp = [
            hp
            for hp in true_function.config_space.get_hyperparameters()
            if hp.name not in selected_hyperparameter_names
        ]

        integral = true_function.pd_integral(*not_selected_hp)  # TODO: Add seed here (from algorithm?)

        for i in range(num_grid_points):
            unscaled_x = unscale_float(self.x_points[0, i, hyperparameter_idx], self.config_space,
                                       self.selected_hyperparameter[0])

            true_y[i] = integral(**{self.selected_hyperparameter[0].name: unscaled_x})

        # regions pdp estimate:
        pdp_y_points = np.mean(self.y_points, axis=0)

        # method == "pdp_sd"
        # pdp_y_std = np.mean(np.sqrt(self.y_variances), axis=0)

        # method != "pdp_sd" (Default in the paper)
        pdp_y_std = np.sqrt(np.mean(self.y_variances, axis=0))

        log_prob = norm.logpdf(true_y, loc=pdp_y_points, scale=pdp_y_std)
        result = - np.mean(log_prob)
        return result

    def delta_nll(self, true_function: BlackboxFunction, full_region: "Region") -> float:
        nll = self.negative_log_likelihood(true_function)
        nll_root = full_region.negative_log_likelihood(true_function)
        return calculate_log_delta(nll, nll_root)

    @cached_property
    def pdp_as_ice_curve(self) -> ICECurve:
        x_pdp = np.mean(self.x_points, axis=0)
        y_pdp = np.mean(self.y_points, axis=0)
        y_variances_pdp = np.mean(self.y_variances, axis=0)
        pdp = ICECurve(
            full_config_space=self.config_space,
            selected_hyperparameter=self.selected_hyperparameter,
            x_ice=x_pdp,
            y_ice=y_pdp,
            y_variances=y_variances_pdp,
            name="PDP in Region"
        )
        return pdp

    def plot_values(self, color="red", ax: Optional[plt.Axes] = None):
        self.pdp_as_ice_curve.plot_values(color=color, ax=ax)

    def plot_confidences(self,
                         line_color="blue",
                         gradient_color="lightblue",
                         confidence_max_sigma: float = 1.5,
                         ax: Optional[plt.Axes] = None):
        self.pdp_as_ice_curve.plot_confidences(
            line_color=line_color,
            gradient_color=gradient_color,
            confidence_max_sigma=confidence_max_sigma,
            ax=ax)

class Partitioner(Algorithm, ABC):
    def __init__(self, surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 samples: np.ndarray,
                 num_grid_points_per_axis: int = 20,
                 not_splittable_hp: Optional[SelectedHyperparameterType] = None,  # more hp to ignore for splitting
                 seed=None):
        super().__init__(
            surrogate_model=surrogate_model,
            selected_hyperparameter=selected_hyperparameter,
            samples=samples,
            num_grid_points_per_axis=num_grid_points_per_axis,
            seed=seed
        )

        # Properties
        self._ice: Optional[ICE] = None

        # save inputs of last calculation to save time
        self.max_depth = Optional[None]

        # get indices of selected hyperparameters
        cs = self.surrogate_model.config_space

        if not_splittable_hp is None:
            self.not_splittable_hp = []
        else:
            self.not_splittable_hp = get_hyperparameters(not_splittable_hp, self.config_space)

        selected_hyperparameter_names = {hp.name for hp in self.selected_hyperparameter}
        selected_hyperparameter_names = selected_hyperparameter_names.union({hp.name for hp in self.not_splittable_hp})
        self.possible_split_parameters: List[CSH.Hyperparameter] = [
            hp for hp in cs.get_hyperparameters()
            if hp.name not in selected_hyperparameter_names
        ]

    @property
    def ice(self) -> ICE:
        if self._ice is None:
            self._ice = ICE(
                surrogate_model=self.surrogate_model,
                selected_hyperparameter=self.selected_hyperparameter,
                samples=self.samples,
                num_grid_points_per_axis=self.num_grid_points_per_axis
            )
        return self._ice

    @abstractmethod
    def partition(self, max_depth: int = 1): # -> List[Region]:
        pass
