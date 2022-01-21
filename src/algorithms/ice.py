from dataclasses import dataclass
from typing import Union, Iterable, Optional, List

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt

from src.algorithms import Algorithm
from src.utils.plotting import Plottable, get_ax, plot_1D_confidence_color_gradients, plot_1D_confidence_lines, \
    plot_line, check_and_set_axis
from src.surrogate_models import SurrogateModel
from src.utils.typing import SelectedHyperparameterType, ColorType
from src.utils.utils import unscale, get_selected_idx


@dataclass
class ICECurve(Plottable):
    def __init__(self,
                 config_space: CS.ConfigurationSpace,
                 selected_hyperparameter: List[CSH.Hyperparameter],
                 x_ice: np.ndarray,
                 y_ice: np.ndarray,
                 y_variances: np.ndarray,
                 name="ICE-Curve"):
        """
        :param x_ice: Shape: (num_gridpoints, num_features) - scaled between [0..1]
        :param y_ice: Shape: (num_gridpoints)
        :param y_variances: (num_gridpoints)
        """
        super().__init__()
        self.config_space = config_space
        self.selected_hyperparameter = selected_hyperparameter
        self.x_ice: np.ndarray = x_ice
        self.y_ice: np.ndarray = y_ice
        self.y_variances: np.ndarray = y_variances
        self.name = name

    def plot(self,
             line_color="red",
             gradient_color="xkcd:light red",
             with_confidence=False,
             ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)
        check_and_set_axis(ax, self.selected_hyperparameter)

        idx = get_selected_idx(self.selected_hyperparameter, self.config_space)
        sigmas = np.sqrt(self.y_variances)
        x_unscaled = unscale(self.x_ice, self.config_space)

        # Switch cases for number of dimensions
        n_hyperparameters = len(self.selected_hyperparameter)
        if n_hyperparameters == 1:  # 1D
            x = x_unscaled[:, idx[0]]
            if with_confidence:
                plot_1D_confidence_color_gradients(x, self.y_ice, sigmas, color=gradient_color, ax=ax)
                plot_1D_confidence_lines(x, self.y_ice, sigmas, k_sigmas=(1, 2),
                                         color=line_color, ax=ax, name=self.name)
            plot_line(x, self.y_ice, color=line_color, label=self.name, ax=ax)

        elif n_hyperparameters == 2:  # 2D
            raise NotImplementedError("2D currently not implemented (#TODO)")
        else:
            raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")

    @property
    def config(self) -> CS.Configuration:
        # TODO: Config (alle fix außer selected hyperparameter)
        return


class ICE(Algorithm):
    def __init__(self,
                 surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 num_samples: int = 1000,
                 num_grid_points_per_axis: int = 20,
                 seed=None):
        super().__init__(surrogate_model, selected_hyperparameter, num_samples, num_grid_points_per_axis, seed=seed)
        self.centered = False  # Can be set directly in class

        # Properties
        self._x_ice: Optional[np.ndarray] = None
        self._y_ice: Optional[np.ndarray] = None
        self._y_variances: Optional[np.ndarray] = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return ICECurve(self.config_space,
                        self.selected_hyperparameter,
                        self.x_ice[idx],
                        self.y_ice[idx],
                        self.y_variances[idx],
                        name=f"ICE-Curve[{idx}]")

    def reset(self):
        """
        Reset all calculations so they can be done again (maybe with different number of samples or centered)
        """
        self._x_ice: Optional[np.ndarray] = None
        self._y_ice: Optional[np.ndarray] = None
        self._y_variances: Optional[np.ndarray] = None

    def _calculate(self):
        self.logger.info("Recalculating ICE...")
        # Retrieve hp index from cs
        cs = self.surrogate_model.config_space
        idx = get_selected_idx(self.selected_hyperparameter, cs)
        num_features = len(cs.get_hyperparameters())

        # retrieve x-values from config
        x = np.asarray([config.get_array() for config in cs.sample_configuration(self.num_samples)])
        x_s = np.linspace(0, 1, self.num_grid_points)

        # TODO: For more than 1 dimension: remove
        x_s = np.expand_dims(x_s, axis=1)

        # create x values by repeating x_s along a new dimension
        x_ice = x.repeat(self.num_grid_points)
        x_ice = x_ice.reshape((self.num_samples, num_features, self.num_grid_points))
        x_ice = x_ice.transpose((0, 2, 1))
        x_ice[:, :, idx] = x_s

        # predictions of surrogate
        means, stds = self.surrogate_model.predict(x_ice.reshape(-1, num_features))
        y_ice = means.reshape((self.num_samples, self.num_grid_points))
        stds = stds.reshape((self.num_samples, self.num_grid_points))
        variances = np.square(stds)

        # center values
        if self.centered:
            y_start = y_ice[:, 0].repeat(self.num_grid_points).reshape(self.num_samples, self.num_grid_points)
            y_ice -= y_start

        assert len(y_ice) == len(variances) == self.num_samples
        self._x_ice = x_ice
        self._y_ice = y_ice
        self._y_variances = variances

        return x_ice, y_ice, variances

    @property
    def x_ice(self) -> np.ndarray:
        if self._x_ice is None:
            self._calculate()
        return self._x_ice

    @property
    def y_ice(self) -> np.ndarray:
        if self._y_ice is None:
            self._calculate()
        return self._y_ice

    @property
    def y_variances(self) -> np.ndarray:
        if self._y_variances is None:
            self._calculate()
        return self._y_variances

    def plot(self,
             color: ColorType = "red",
             alpha=0.1,
             ax: Optional[plt.Axes] = None):
        # Resolve arguments
        ax = get_ax(ax)
        check_and_set_axis(ax, self.selected_hyperparameter)

        # Plot
        if self.n_selected_hyperparameter == 1:  # 1D
            x_ice = unscale(self.x_ice, self.surrogate_model.config_space)
            y_ice = self.y_ice
            idx = self.surrogate_model.config_space.get_idx_by_hyperparameter_name(self.selected_hyperparameter[0].name)
            ax.plot(x_ice[:, :, idx].T, y_ice.T, alpha=alpha, color=color)
            ax.plot([], [], color=color, label="ICE")  # Hacky label for plot...
        elif self.n_selected_hyperparameter == 2:  # 2D
            raise NotImplementedError("2D currently not implemented (#TODO)")
        else:
            raise NotImplementedError(f"Plotting for {self.n_selected_hyperparameter} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")
