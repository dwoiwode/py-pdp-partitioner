from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, Iterable

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from pyPDP.surrogate_models import SurrogateModel
from pyPDP.utils.plotting import get_ax, check_and_set_axis
from pyPDP.utils.utils import get_hyperparameters, get_selected_idx, ConfigSpaceHolder


class AcquisitionFunction(ConfigSpaceHolder, ABC):
    def __init__(self,
                 config_space: CS.ConfigurationSpace,
                 surrogate_model: SurrogateModel,
                 samples_for_optimization: int = 100,
                 minimize_objective: bool = True,
                 seed=None):
        super().__init__(config_space, seed=seed)
        self.surrogate_model = surrogate_model
        self.n_samples_for_optimization = samples_for_optimization
        self.minimize_objective = minimize_objective

    @abstractmethod
    def __call__(self, configuration: CS.Configuration) -> Union[float, np.ndarray]:
        pass

    def update(self, eta: float):
        pass

    def get_optimum(self) -> CS.Configuration:
        return self._get_optimum_uniform_distribution()[0]

    def _get_optimum_uniform_distribution(self) -> Tuple[CS.Configuration, float]:
        configs = self.config_space.sample_configuration(self.n_samples_for_optimization)
        values = self(configs)
        config_value_pairs = [(config, value) for config, value in zip(configs, values)]

        return max(config_value_pairs, key=lambda x: x[1])

    def convert_configs(self, configuration: Union[CS.Configuration, np.ndarray]):
        if isinstance(configuration, CS.Configuration):
            x = np.asarray(configuration.get_array())
            x = x.reshape([1, -1])
        elif isinstance(configuration, list):
            x = []
            for config in configuration:
                if isinstance(config, CS.Configuration):
                    x.append(config.get_array())
                else:
                    x.append(config.copy())
            x = np.asarray(x)
        else:
            x = configuration.copy()
        return x

    def plot(self,
             color_acquisition="darkgreen",
             color_optimum="red",
             show_optimum=True,
             x_hyperparameters: Optional[Iterable[CSH.Hyperparameter]] = None,
             ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)
        x_hyperparameters = get_hyperparameters(x_hyperparameters, self.config_space)
        check_and_set_axis(ax, x_hyperparameters, ylabel="Acquisition")

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

            if show_optimum:
                ax.plot(list(optimum.values())[0], self(optimum), "*", color=color_optimum, label=f"Optimum ({optimum})",
                        markersize=15)
        elif n_hyperparameters == 2:  # 2D
            idx = get_selected_idx(x_hyperparameters, self.config_space)
            raise NotImplementedError("2D currently not implemented (#TODO)")
        else:
            raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")


class ExpectedImprovement(AcquisitionFunction):
    def __init__(
            self,
            config_space,
            surrogate_model: SurrogateModel,
            eps: float = 0.0,  # Exploration parameter
            samples_for_optimization=100,
            minimize_objective=True,
            seed=None
    ):
        super().__init__(
            config_space,
            surrogate_model,
            samples_for_optimization,
            minimize_objective, seed=seed
        )
        if not minimize_objective:
            raise NotImplementedError('EI for maximization')
        self.eta = 0
        self.exploration = eps

    def __call__(self, configuration: Union[CS.Configuration, np.ndarray]) -> Union[float, np.ndarray]:
        x = self.convert_configs(configuration)

        mean, sigma = self.surrogate_model.predict(x)

        Z = (self.eta - mean - self.exploration) / sigma
        Phi_Z = norm.cdf(Z)
        phi_Z = norm.pdf(Z)
        ret = sigma * (Z * Phi_Z + phi_Z)
        ret[sigma == 0] = 0
        return ret

    def update(self, eta: float):
        self.eta = eta


class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self,
                 config_space: CS.ConfigurationSpace,
                 surrogate_model: SurrogateModel,
                 eps: float = 0.1,  # Exploration parameter
                 samples_for_optimization: int = 100,
                 minimize_objective=True,
                 seed=None):
        super().__init__(config_space, surrogate_model, samples_for_optimization=samples_for_optimization,
                         minimize_objective=minimize_objective, seed=seed)
        self.eta = 0
        self.exploration = eps

    def __call__(self, configuration: Union[CS.Configuration, np.ndarray]) -> Union[float, np.ndarray]:
        x = self.convert_configs(configuration)

        mean, sigma = self.surrogate_model.predict(x)

        if self.minimize_objective:
            temp = (self.eta - mean - self.exploration) / sigma
        else:
            temp = (mean - self.eta - self.exploration) / sigma
        prob_of_improvement = norm.cdf(temp)
        prob_of_improvement[sigma == 0] = 0
        return prob_of_improvement

    def update(self, eta: float):
        self.eta = eta


class LowerConfidenceBound(AcquisitionFunction):
    """LCB"""
    def __init__(self,
                 config_space: CS.ConfigurationSpace,
                 surrogate_model: SurrogateModel,
                 tau: float = 5,
                 samples_for_optimization=100,
                 minimize_objective=True,
                 seed=None):
        super().__init__(config_space, surrogate_model, samples_for_optimization, minimize_objective=minimize_objective,
                         seed=seed)
        self.tau = tau

    def __call__(self, configuration: Union[CS.Configuration, np.ndarray]) -> Union[float, np.ndarray]:
        x = self.convert_configs(configuration)

        mean, sigma = self.surrogate_model.predict(x)
        if self.minimize_objective:
            return - mean + self.tau * sigma
        else:
            return mean + self.tau * sigma
