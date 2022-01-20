import hashlib
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Any, List, Tuple, Optional, Union, Iterable

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.surrogate_models import SurrogateModel, GaussianProcessSurrogate
from src.utils.plotting import Plottable, get_ax
from src.utils.utils import config_list_to_2d_arr, get_hyperparameters, get_uniform_distributed_ranges, get_selected_idx
from src.utils.typing import ColorType


class Sampler(Plottable, ABC):
    def __init__(self,
                 obj_func: Callable,
                 config_space: CS.ConfigurationSpace,
                 minimize_objective=True,
                 seed=None):
        super().__init__()
        self.obj_func = obj_func
        self.config_space = config_space
        self.minimize_objective = minimize_objective
        self.seed = seed

        self.config_list: List[CS.Configuration] = []
        self.y_list: List[float] = []

    def __len__(self) -> int:
        return len(self.config_list)

    def reset(self):
        self.config_list = []
        self.y_list = []

    @property
    def X(self) -> np.ndarray:
        return config_list_to_2d_arr(self.config_list)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self.y_list)

    @property
    def incumbent(self) -> Tuple[Optional[CS.Configuration], float]:
        if len(self.y_list) == 0:
            return None, float("inf")
        if self.minimize_objective:
            incumbent_index = np.argmin(self.y_list)
        else:
            incumbent_index = np.argmax(self.y_list)

        incumbent_config = self.config_list[incumbent_index]
        incumbent_value = self.y_list[incumbent_index]
        return incumbent_config, incumbent_value

    @abstractmethod
    def sample(self, n_points: int = 1):
        """ Samples n_points new points """
        pass

    def plot(self,
             color: ColorType = "red",
             marker: str = ".",
             label: Optional[str] = None,
             *,  # Prevent from next args to be added via positional arguments
             x_hyperparameters: Optional[Iterable[CSH.Hyperparameter]] = None,
             ax: Optional[plt.Axes] = None):
        # Resolve arguments
        ax = get_ax(ax)
        x_hyperparameters = get_hyperparameters(x_hyperparameters, self.config_space)
        if label is None:
            label = f"Sampled points ({self.__class__.__name__})"

        # Check whether plot is possible
        # TODO: New method: Check whether labels fit with new plotting labels and add if missing
        # x_label
        # y_label

        # Plot
        plotting_kwargs = {
            "marker": marker,
            "linestyle": "",
            "color": color,
            "label": label
        }

        n_hyperparameters = len(x_hyperparameters)
        if n_hyperparameters == 1:  # 1D
            hp = x_hyperparameters[0]
            x = np.asarray([config[hp.name] for config in self.config_list])
            order = np.argsort(x)
            ax.plot(x[order], self.y[order], **plotting_kwargs)
        elif n_hyperparameters == 2:  # 2D
            hp1, hp2 = x_hyperparameters
            x1, x2 = zip(*[(config[hp1], config[hp2]) for config in self.config_list])
            colors = self.y  # TODO: How to plot values
            ax.scatter(x1, x2, c=colors, **plotting_kwargs)
        else:
            raise NotImplemented("Plotting for more than 2 dimensions not implemented. "
                                 "Please select a specific hp by setting `x_hyperparemeters`")


class GridSampler(Sampler):
    pass


class RandomSampler(Sampler):
    """
    Seed has to be already used for configuration space for
    """

    def __init__(self,
                 obj_func: Callable,
                 config_space: CS.ConfigurationSpace,
                 minimize_objective=True,
                 seed=None):
        super().__init__(obj_func, config_space, minimize_objective=minimize_objective, seed=seed)
        self.rng = np.random.RandomState(self.seed)

    def sample(self, n_points: int = 1):
        n_features = len(self.config_space.get_hyperparameters())
        samples = self.rng.random((n_points, n_features))
        origin = self.__class__.__name__
        for i in range(n_points):
            config = CS.Configuration(self.config_space, vector=samples[i], origin=origin)
            value = self.obj_func(**config)
            self.config_list.append(config)
            self.y_list.append(value)


class BayesianOptimizationSampler(Sampler):
    def __init__(self,
                 obj_func: Callable[[Any], float],
                 config_space: CS.ConfigurationSpace,
                 surrogate_model=None,
                 initial_points: int = 5,
                 acq_class=None,
                 acq_class_kwargs=None,
                 minimize_objective: bool = True,
                 seed=None):
        super().__init__(obj_func, config_space, minimize_objective, seed=seed)
        self.initial_points = initial_points  # number of initial points to be sampled

        # Surrogate model
        if surrogate_model is None:
            surrogate_model = GaussianProcessSurrogate(self.config_space, seed=seed)
        self.surrogate_model = surrogate_model
        self._model_fitted_hash: str = ""

        # Acquisition function
        if acq_class_kwargs is None:
            acq_class_kwargs = {}
        if acq_class is None:
            acq_class = LowerConfidenceBound  # Default Lower Confidence Bound
        self.acq_func: AcquisitionFunction = acq_class(self.config_space,
                                                       self.surrogate_model,
                                                       minimize_objective=minimize_objective,
                                                       seed=seed,
                                                       **acq_class_kwargs)

    def _sample_initial_points(self, max_sampled_points=None):
        if max_sampled_points is None:
            sampled_points = self.initial_points
        else:
            sampled_points = min(self.initial_points, max_sampled_points)

        self.config_list = self.config_space.sample_configuration(sampled_points)
        if self.initial_points == 1:  # for a single value, the sampling does not return a list
            self.config_list = [self.config_list]

        self.y_list = [self.obj_func(**config) for config in self.config_list]
        self.fit_surrogate(force=True)

    def fit_surrogate(self, force: bool = False):
        """
        Fits the surrogate model. If force is False and surrogate model already fitted with current configs, do nothing
        """
        parameter_hash = hashlib.md5()
        parameter_hash.update(str(self.config_list).encode("latin"))
        if force or self._model_fitted_hash != parameter_hash:
            self.surrogate_model.fit(self.X, self.y_list)

        self._model_fitted_hash = parameter_hash

    def sample(self, n_points: int = 1):
        # Sample initial random points if not already done or given
        already_sampled = 0
        current_points = len(self)
        if current_points < self.initial_points:
            self._sample_initial_points(n_points)
            already_sampled = len(self) - current_points

        # Update surrogate model
        self.fit_surrogate()

        for i in range(n_points - already_sampled):
            # select next point
            self.acq_func.update(self.incumbent[1])
            new_best_candidate = self.acq_func.get_optimum()
            new_y = self.obj_func(**new_best_candidate)

            # add new point
            self.config_list.append(new_best_candidate)
            self.y_list.append(new_y)

            # Update surrogate model
            self.fit_surrogate()

    def surrogate_score(self, configs: Union[np.ndarray, List[CS.Configuration]]) -> Tuple[np.ndarray, np.ndarray]:
        if len(configs) == 0:
            warnings.warn("No configs provided. Returning empty surrogate scores")
            return np.asarray([]), np.asarray([])

        if isinstance(configs, list) and isinstance(configs[0], CS.Configuration):
            configs = [config.get_array() for config in configs]

        x = np.asarray(configs)
        return self.surrogate_model.predict(x, return_std=True)


class AcquisitionFunction(Plottable, ABC):
    def __init__(self,
                 config_space: CS.ConfigurationSpace,
                 surrogate_model: SurrogateModel,
                 samples_for_optimization: int = 100,
                 minimize_objective: bool = True,
                 seed=None):
        super().__init__()
        self.surrogate_model = surrogate_model
        self.config_space = config_space
        self.n_samples_for_optimization = samples_for_optimization
        self.minimize_objective = minimize_objective
        self.seed = seed

    @abstractmethod
    def __call__(self, configuration: CS.Configuration) -> float:
        pass

    def update(self, eta: float):
        pass

    def get_optimum(self) -> CS.Configuration:
        return self._get_optimum_uniform_distribution()[0]

    def _get_optimum_uniform_distribution(self) -> Tuple[CS.Configuration, float]:
        config_value_pairs = []
        for config in self.config_space.sample_configuration(self.n_samples_for_optimization):
            config_value_pairs.append((config, self(config.get_array())))

        return max(config_value_pairs, key=lambda x: x[1])

    def convert_configs(self, configuration: Union[CS.Configuration, np.ndarray]):
        if isinstance(configuration, CS.Configuration):
            x = np.asarray(configuration.get_array())
        else:
            x = configuration.copy()
        x = x.reshape([1, -1])
        return x

    def plot(self,
             color_acquisition="darkgreen",
             color_optimum="red",
             show_optimum=True,
             x_hyperparameters: Optional[Iterable[CSH.Hyperparameter]] = None,
             ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)
        x_hyperparameters = get_hyperparameters(x_hyperparameters, self.config_space)

        # Sample configs and get values of acquisition function
        configs = self.config_space.sample_configuration(self.n_samples_for_optimization * len(x_hyperparameters))
        acquisition_y = np.asarray([self(x) for x in configs]).reshape(-1)
        x = np.asarray([[config[hp.name] for hp in x_hyperparameters] for config in configs])

        # Get optimum
        optimum = self.get_optimum()

        # Plot
        n_hyperparameters = len(tuple(x_hyperparameters))
        if n_hyperparameters == 1:  # 1D
            # Sort by x axis
            order = np.argsort(x, axis=0)[:, 0]
            x = x[order, 0]
            acquisition_y = acquisition_y[order]

            ax.fill_between(x, acquisition_y, color=color_acquisition, alpha=0.3)
            ax.plot(x, acquisition_y, color=color_acquisition, label=self.__class__.__name__)

            ax.plot(list(optimum.values())[0], self(optimum), "*", color=color_optimum, label=f"Optimum ({optimum})",
                    markersize=15)
        elif n_hyperparameters == 2:  # 2D
            idx = get_selected_idx(x_hyperparameters, self.config_space)
            raise NotImplemented("2D currently not implemented (#TODO)")
        else:
            raise NotImplemented("Plotting for more than 2 dimensions not implemented. "
                                 "Please select a specific hp by setting `x_hyperparemeters`")


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self,
                 config_space,
                 surrogate_model: SurrogateModel,
                 eps: float=0.0,  # Exploration parameter
                 samples_for_optimization=100,
                 minimize_objective=True,
                 seed=None):
        super().__init__(config_space,
                         surrogate_model,
                         samples_for_optimization,
                         minimize_objective, seed=seed)
        self.eta = 0
        self.exploration = eps

    def __call__(self, configuration: Union[CS.Configuration, np.ndarray]):
        x = self.convert_configs(configuration)

        mean, sigma = self.surrogate_model.predict(x)
        if sigma == 0:
            return 0

        Z = (self.eta - mean - self.exploration) / sigma
        Phi_Z = norm.cdf(Z)
        phi_Z = norm.pdf(Z)
        return sigma * (Z * Phi_Z + phi_Z)

    def update(self, eta: float):
        self.eta = eta


class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self,
                 config_space: CS.ConfigurationSpace,
                 surrogate_model: SurrogateModel,
                 eps: float = 0.0,  # Exploration parameter
                 samples_for_optimization: int = 100,
                 minimize_objective=True,
                 seed=None):
        super().__init__(config_space, surrogate_model, samples_for_optimization=samples_for_optimization,
                         minimize_objective=minimize_objective, seed=seed)
        self.eta = 0
        self.exploration = eps

    def __call__(self, configuration: Union[CS.Configuration, np.ndarray]):
        x = self.convert_configs(configuration)

        mean, sigma = self.surrogate_model.predict(x)
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


class LowerConfidenceBound(AcquisitionFunction):
    def __init__(self,
                 config_space: CS.ConfigurationSpace,
                 surrogate_model: SurrogateModel,
                 theta: float = 5,
                 samples_for_optimization=100,
                 minimize_objective=True,
                 seed=None):
        super().__init__(config_space, surrogate_model, samples_for_optimization, minimize_objective=minimize_objective,
                         seed=seed)
        self.theta = theta

    def __call__(self, configuration: Union[CS.Configuration, np.ndarray]):
        x = self.convert_configs(configuration)

        mean, sigma = self.surrogate_model.predict(x)
        if self.minimize_objective:
            return - mean + self.theta * sigma
        else:
            return mean + self.theta * sigma
