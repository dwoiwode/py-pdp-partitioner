import abc
from typing import Callable, Any

import ConfigSpace as CS
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from src.utils import config_list_to_2d_arr


class AbstractOptimizer(abc.ABC):
    def __init__(self, obj_func: Callable, config_space: CS.ConfigurationSpace, minimize_objective=True):
        self.obj_func = obj_func
        self.config_space = config_space
        self.values: dict[CS.Configuration, float] = {}
        self.minimize_objective = minimize_objective

    @property
    def incumbent(self) -> tuple[CS.Configuration, float]:
        if self.minimize_objective:
            incumbent = min(self.values, key=self.values.get)
        else:
            incumbent = max(self.values, key=self.values.get)
        return incumbent, self.values[incumbent]

    @abc.abstractmethod
    def optimize(self, n_points: int = 1) -> CS.Configuration:
        pass


class GridSearch(AbstractOptimizer):
    pass


class RandomSearch(AbstractOptimizer):
    def optimize(self, n_points: int = 1) -> CS.Configuration:
        for i in range(n_points):
            config = self.config_space.sample_configuration()
            value = self.obj_func(**config)
            self.values[config] = value

        return self.incumbent[0]


class BayesianOptimization(AbstractOptimizer):
    def __init__(self, obj_func: Callable[[Any], float], config_space: CS.ConfigurationSpace,
                 initial_points: int = 5, config_list: list[CS.Configuration] = None, y_list: list[float] = None,
                 minimize_objective: bool = True, eps: float = 0.1):
        super().__init__(obj_func, config_space, minimize_objective)
        self.model = GaussianProcessRegressor()  # surrogate model
        self.eps = eps  # exploration factor for acq-function
        self.acq_sample_num = 100  # number of points sampled during acquisition function decision of next point
        self.acq_func = self._probability_of_improvement

        assert initial_points > 0 or (len(config_list) > 0 and len(y_list) > 0), \
            'At least one initial random point is required'
        self.initial_points = initial_points  # number of initial points to be sampled
        self.config_list = config_list
        self.y_list = y_list

    def _sample_initial_points(self):
        self.config_list = self.config_space.sample_configuration(self.initial_points)
        if self.initial_points == 1:  # for a single value, the sampling does not return a list
            self.config_list = [self.config_list]
        x_arr_list = config_list_to_2d_arr(self.config_list)
        self.y_list = [self.obj_func(**config) for config in self.config_list]
        self.model.fit(x_arr_list, self.y_list)

    def optimize(self, n_points: int = 1):
        # sample initial random points if not already done or given
        if self.config_list is None or self.y_list is None:
            self._sample_initial_points()

        for i in range(n_points):
            # select next point
            best_config = self.acq_func()
            new_y = self.obj_func(**best_config)

            # add new point
            self.config_list.append(best_config)
            self.y_list.append(new_y)

            # update surrogate model
            x_arr_list = config_list_to_2d_arr(self.config_list)
            self.model.fit(x_arr_list, self.y_list)

        # find best configuration so far
        if self.minimize_objective:
            best_idx = np.argmin(self.y_list)
        else:
            best_idx = np.argmax(self.y_list)
        best_config = self.config_list[best_idx]
        return best_config

    def _probability_of_improvement(self) -> CS.Configuration:
        # sample points
        x_sample_config = self.config_space.sample_configuration(self.acq_sample_num)
        x_sample_array = config_list_to_2d_arr(x_sample_config)
        means, stds = self.surrogate_score(x_sample_array)

        # prob of improvement for sampled points
        if self.minimize_objective:
            cur_best_y = np.min(self.y_list)
            temp = - (means - cur_best_y - self.eps) / stds
        else:
            cur_best_y = np.max(self.y_list)
            temp = (means - cur_best_y - self.eps) / stds
        prob_of_improvement = norm.cdf(temp)

        # best sampled point
        best_idx = np.argmax(prob_of_improvement)
        best_new_config = x_sample_config[best_idx]
        return best_new_config

    def surrogate_score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x)
        assert len(x.shape) == 1 or len(x.shape) == 2, 'Can only compute surrogate score for 1d or 2d arrays'
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)
        return self.model.predict(x, return_std=True)
