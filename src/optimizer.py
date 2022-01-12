import abc
import hashlib
import warnings
from typing import Callable, Any, List, Dict, Tuple, Optional, Union, Type

import ConfigSpace as CS
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import config_list_to_2d_arr


class AbstractOptimizer(abc.ABC):
    def __init__(self, obj_func: Callable, config_space: CS.ConfigurationSpace, minimize_objective=True):
        self.obj_func = obj_func
        self.config_space = config_space
        self.config_list: List[CS.Configuration] = []
        self.y_list: List[float] = []
        self.minimize_objective = minimize_objective

    @property
    def incumbent(self) -> Tuple[Optional[CS.Configuration], float]:
        if len(self.y_list) == 0:
            return None, 10000
        if self.minimize_objective:
            incumbent_index = np.argmin(self.y_list)
        else:
            incumbent_index = np.argmax(self.y_list)

        incumbent_config = self.config_list[incumbent_index]
        incumbent_value = self.y_list[incumbent_index]
        return incumbent_config, incumbent_value

    @abc.abstractmethod
    def optimize(self, n_points: int = 1) -> CS.Configuration:
        pass

    @abc.abstractmethod
    def surrogate_score(self, configs: List[CS.Configuration]) -> Tuple[np.ndarray, np.ndarray]:
        pass


class GridSearch(AbstractOptimizer):
    pass


class RandomSearch(AbstractOptimizer):
    def optimize(self, n_points: int = 1) -> CS.Configuration:
        for i in range(n_points):
            config = self.config_space.sample_configuration()
            value = self.obj_func(**config)
            self.config_list.append(config)
            self.y_list.append(value)

        return self.incumbent[0]


class BayesianOptimization(AbstractOptimizer):
    def __init__(self, obj_func: Callable[[Any], float], config_space: CS.ConfigurationSpace,
                 initial_points: int = 5, config_list: List[CS.Configuration] = None, y_list: List[float] = None,
                 minimize_objective: bool = True, eps: float = 0.1):
        super().__init__(obj_func, config_space, minimize_objective)
        self.model = Pipeline([
            ("standardize", StandardScaler()),
            ("GP", GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True,
                                            n_restarts_optimizer=10,
                                            random_state=0)),
        ])
        self.eps = eps  # exploration factor for acq-function
        self.acq_sample_num = 100  # number of points sampled during acquisition function decision of next point
        self.acq_func: AcquisitionFunction = ProbabilityOfImprovement(self.config_space, self.model,
                                                                      minimize_objective=minimize_objective)

        assert initial_points > 0 or (len(config_list) > 0 and len(y_list) > 0), \
            'At least one initial random point is required'
        self.initial_points = initial_points  # number of initial points to be sampled
        self.config_list = config_list
        self.y_list = y_list

        self._model_fitted_hash: str = ""

    def _sample_initial_points(self):
        self.config_list = self.config_space.sample_configuration(self.initial_points)
        if self.initial_points == 1:  # for a single value, the sampling does not return a list
            self.config_list = [self.config_list]

        self.y_list = [self.obj_func(**config) for config in self.config_list]
        self.fit_surrogate()

    def fit_surrogate(self):
        parameter_hash = hashlib.md5()
        parameter_hash.update(str(self.config_list).encode("latin"))
        if self._model_fitted_hash != parameter_hash:
            x_arr_list = config_list_to_2d_arr(self.config_list)
            self.model.fit(x_arr_list, self.y_list)

        self._model_fitted_hash = parameter_hash

    def optimize(self, n_points: int = 1):
        # sample initial random points if not already done or given
        if self.config_list is None or self.y_list is None:
            self._sample_initial_points()

        # Update surrogate model
        self.fit_surrogate()

        for i in range(n_points):
            # select next point
            self.acq_func.update(self.incumbent[1])
            new_best_candidate = self.acq_func.get_optimum()
            new_y = self.obj_func(**new_best_candidate)

            # add new point
            self.config_list.append(new_best_candidate)
            self.y_list.append(new_y)

            # Update surrogate model
            self.fit_surrogate()

        # return best configuration so far
        return self.incumbent[0]

    def surrogate_score(self, configs: Union[np.ndarray, List[CS.Configuration]]) -> Tuple[np.ndarray, np.ndarray]:
        if len(configs) == 0:
            warnings.warn("No configs provided. Returning empty surrogate scores")
            return np.asarray([]), np.asarray([])

        if isinstance(configs, list) and isinstance(configs[0], CS.Configuration):
            configs = [config.get_array() for config in configs]

        x = np.asarray(configs)
        return self.model.predict(x, return_std=True)


class AcquisitionFunction(abc.ABC):
    def __init__(self, config_space: CS.ConfigurationSpace, samples=100, minimize_objective=True):
        self.config_space = config_space
        self.samples = samples
        self.minimize_objective = minimize_objective

    @abc.abstractmethod
    def __call__(self, configuration: CS.Configuration) -> float:
        pass

    def update(self, eta: float):
        pass

    def get_optimum(self) -> CS.Configuration:
        return self._get_optimum_uniform_distribution()[0]

    def _get_optimum_uniform_distribution(self) -> Tuple[CS.Configuration, float]:
        config_value_pairs = []
        for config in self.config_space.sample_configuration(self.samples):
            config_value_pairs.append((config, self(config.get_array())))

        # if self.minimize_objective:
        #     return min(config_value_pairs, key=lambda x: x[1])
        # else:
        return max(config_value_pairs, key=lambda x: x[1])


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, config_space,
                 surrogate_model: Union[GaussianProcessRegressor, Pipeline],
                 samples=100, minimize_objective=True):
        super().__init__(config_space, samples=samples, minimize_objective=minimize_objective)
        self.surrogate_model = surrogate_model
        self.eta = 0
        self.exploration = 0  # Exploration parameter

    def __call__(self, configuration: Union[CS.Configuration, np.ndarray]):
        # print(type(configuration), inspect.getouterframes(inspect.currentframe(), 2)[2][3])
        if isinstance(configuration, CS.Configuration):
            x = np.asarray(configuration.get_array())
        else:
            x = configuration.copy()
        x = x.reshape([1, -1])

        mean, sigma = self.surrogate_model.predict(x, return_std=True)
        if sigma == 0:
            return 0

        Z = (self.eta - mean - self.exploration) / sigma
        Phi_Z = norm.cdf(Z)
        phi_Z = norm.pdf(Z)
        return sigma * (Z * Phi_Z + phi_Z)

    def update(self, eta: float):
        self.eta = eta


class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, config_space: CS.ConfigurationSpace,
                 surrogate_model: Union[GaussianProcessRegressor, Pipeline],
                 samples=100, minimize_objective=True):
        super().__init__(config_space, samples=samples, minimize_objective=minimize_objective)
        self.surrogate_model = surrogate_model
        self.eta = 0
        self.exploration = 0.0  # Exploration parameter

    def __call__(self, configuration: Union[CS.Configuration, np.ndarray]):
        # print(type(configuration), inspect.getouterframes(inspect.currentframe(), 2)[2][3])
        if isinstance(configuration, CS.Configuration):
            x = np.asarray(configuration.get_array())
        else:
            x = configuration.copy()
        x = x.reshape([1, -1])

        mean, sigma = self.surrogate_model.predict(x, return_std=True)
        if sigma == 0:
            return 0

        if self.minimize_objective:
            temp = (self.eta - mean - self.exploration) / sigma
        else:
            temp = (mean - self.eta - self.exploration) / sigma
        prob_of_improvement = norm.cdf(temp)
        return prob_of_improvement

    def update(self, eta: float):
        self.eta = eta
