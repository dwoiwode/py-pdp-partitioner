from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, Iterable

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from src.surrogate_models import SurrogateModel
from src.utils.plotting import Plottable, get_ax, check_and_set_axis
from src.utils.utils import get_hyperparameters, get_selected_idx


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
        check_and_set_axis(ax, x_hyperparameters)

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
            raise NotImplementedError("2D currently not implemented (#TODO)")
        else:
            raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self,
                 config_space,
                 surrogate_model: SurrogateModel,
                 eps: float = 0.0,  # Exploration parameter
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
