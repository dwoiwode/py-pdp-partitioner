from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional

import ConfigSpace as CS
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.plotting import plot_1D_confidence_lines, plot_1D_confidence_color_gradients, get_ax, \
    plot_line, check_and_set_axis
from src.utils.utils import get_uniform_distributed_ranges, config_list_to_array, ConfigSpaceHolder


class SurrogateModel(ConfigSpaceHolder, ABC):
    def __init__(self, cs: CS.ConfigurationSpace, seed=None):
        super().__init__(cs, seed=seed)
        self.num_fitted_points = 0

    def __call__(self,
                 X: Union[np.ndarray, CS.Configuration, List[CS.Configuration]]
                 ) -> Union[np.ndarray, float, List[float]]:
        # Config or List[Config] or empty list
        if isinstance(X, CS.Configuration):
            means, stds = self.predict_config(X)
        elif isinstance(X, list) and (len(X) == 0) or isinstance(X[0], CS.Configuration):
            # Returns either float or List[float], depending on whether a single config or list of configs is given
            means, stds = self.predict_configs(X)
        elif isinstance(X, np.ndarray):
            # np.ndarray
            means, stds = self.predict(X)[0]
        else:
            raise TypeError(f"Could not interpret {type(X)}")
        return means

    def __str__(self):
        return f"{self.__class__.__name__}(fitted on {self.num_fitted_points} points)"

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and sigma of predictions

        :param X: input data shaped [NxK] for N samples and K parameters
        :return: means, stds
        """
        pass

    def fit(self, X: Union[List[CS.Configuration], np.ndarray], y: Union[List[float], np.ndarray]):
        X = config_list_to_array(X)
        self.num_fitted_points = len(y)
        return self._fit(X, np.asarray(y))

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict_configs(self,
                        configs: List[CS.Configuration]) -> Tuple[List[float], List[float]]:
        """
        If configs is a single config: Return a single mean, std.
        If configs is a list of configs: Return a tuple with list of means and list of stds
        """
        X = config_list_to_array(configs)
        y = self.predict(X)
        means = y[0].tolist()
        stds = y[1].tolist()
        return means, stds

    def predict_config(self, config: CS.Configuration) -> Tuple[float, float]:
        # Single config
        mean, std = self.predict(config.get_array())
        assert isinstance(mean, float)
        assert isinstance(std, float)
        return mean, std

    def plot(self,
             line_color="blue",
             gradient_color="lightblue",
             with_confidence=True,
             samples_per_axis=100,
             ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)

        hyperparameters = self.config_space.get_hyperparameters()
        n_hyperparameters = len(hyperparameters)
        assert n_hyperparameters < 3, 'Surrogate model only supports plotting less than 3 feature dimensions'

        check_and_set_axis(ax, hyperparameters)

        # Switch cases for number of dimensions
        if n_hyperparameters == 1:  # 1D
            ranges = get_uniform_distributed_ranges(self.config_space, samples_per_axis, scaled=False)
            mu, std = self.predict(np.reshape(np.linspace(0, 1, samples_per_axis), (-1, 1)))

            name = self.__class__.__name__
            x = ranges[0]
            if with_confidence:
                plot_1D_confidence_color_gradients(x, mu, std, color=gradient_color, ax=ax)
                plot_1D_confidence_lines(x, mu, std, k_sigmas=(1, 2), color=line_color, ax=ax, name=name)
            plot_line(x, mu, color=line_color, label=f"{name}-$\mu$", ax=ax)
        elif n_hyperparameters == 2:  # 2D
            raise NotImplementedError("2D currently not implemented (#TODO)")
        else:
            raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")


class SkLearnPipelineSurrogate(SurrogateModel):
    def __init__(self, pipeline: Pipeline, cs: CS.ConfigurationSpace, seed=None):
        super().__init__(cs, seed=seed)
        self.pipeline = pipeline

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and sigma of predictions

        :param X: input data shaped [NxK] for N samples and K parameters
        :return: means, sigmas
        """

        return self.pipeline.predict(X, return_std=True)

class GaussianProcessSurrogate(SkLearnPipelineSurrogate):
    def __init__(self, cs: CS.ConfigurationSpace, kernel=Matern(nu=2.5), seed=None):
        pipeline = Pipeline([
            ("standardize", StandardScaler()),
            ("GP", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                            n_restarts_optimizer=20,
                                            alpha=1e-8,
                                            random_state=seed)),
        ])
        super().__init__(pipeline, cs, seed=seed)
