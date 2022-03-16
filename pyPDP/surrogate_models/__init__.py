from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional

import ConfigSpace as CS
import numpy as np
from matplotlib import pyplot as plt

from pyPDP.utils.plotting import get_ax, check_and_set_axis, plot_1D_confidence_color_gradients, plot_1D_confidence_lines, \
    plot_line
from pyPDP.utils.typing import ColorType
from pyPDP.utils.utils import ConfigSpaceHolder, config_list_to_array, get_uniform_distributed_ranges


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

    def plot_means(
            self,
            color: ColorType = "blue",
            samples_per_axis: int = 100,
            ax: Optional[plt.Axes] = None
    ):
        ax = get_ax(ax)

        hyperparameters = self.config_space.get_hyperparameters()
        n_hyperparameters = len(hyperparameters)
        assert n_hyperparameters < 3, 'Surrogate model only supports plotting less than 3 feature dimensions'

        check_and_set_axis(ax, hyperparameters)

        # Switch cases for number of dimensions
        ranges = get_uniform_distributed_ranges(self.config_space, samples_per_axis, scaled=False)
        linspace = np.linspace(0, 1, samples_per_axis)
        if n_hyperparameters == 1:  # 1D
            mu, std = self.predict(np.reshape(linspace, (-1, 1)))

            name = self.__class__.__name__
            x = ranges[0]
            plot_line(x, mu, color=color, label=f"{name}-$\mu$", ax=ax)
        elif n_hyperparameters == 2:  # 2D
            x = ranges[0]
            y = ranges[1]
            xx, yy = np.meshgrid(linspace, linspace)
            mu, std = self.predict(np.asarray([xx, yy]).T.reshape(-1, 2))

            ax.pcolormesh(x, y, mu.reshape(xx.shape).T, shading='auto')
        else:
            raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")

    def plot_confidences(
            self,
            line_color: ColorType = "blue",
            gradient_color: ColorType = "lightblue",
            samples_per_axis: int = 100,
            ax: Optional[plt.Axes] = None
    ):
        ax = get_ax(ax)

        hyperparameters = self.config_space.get_hyperparameters()
        n_hyperparameters = len(hyperparameters)
        assert n_hyperparameters < 3, 'Surrogate model only supports plotting less than 3 feature dimensions'

        check_and_set_axis(ax, hyperparameters)

        # Switch cases for number of dimensions
        ranges = get_uniform_distributed_ranges(self.config_space, samples_per_axis, scaled=False)
        linspace = np.linspace(0, 1, samples_per_axis)
        if n_hyperparameters == 1:  # 1D
            mu, std = self.predict(np.reshape(linspace, (-1, 1)))

            name = self.__class__.__name__
            x = ranges[0]
            plot_1D_confidence_color_gradients(x, mu, std, color=gradient_color, ax=ax)
            plot_1D_confidence_lines(x, mu, std, k_sigmas=(1, 2), color=line_color, ax=ax, name=name)
        elif n_hyperparameters == 2:  # 2D
            x = ranges[0]
            y = ranges[1]
            xx, yy = np.meshgrid(linspace, linspace)
            mu, std = self.predict(np.asarray([xx, yy]).T.reshape(-1, 2))

            ax.pcolormesh(x, y, std.reshape(xx.shape).T, shading='auto')
        else:
            raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")
