from dataclasses import dataclass
from functools import cached_property
from typing import Optional, List

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt

from pyPDP.algorithms import Algorithm
from pyPDP.surrogate_models import SurrogateModel
from pyPDP.utils.plotting import get_ax, plot_1D_confidence_color_gradients, plot_1D_confidence_lines, \
    plot_line, check_and_set_axis, plot_2D
from pyPDP.utils.typing import SelectedHyperparameterType, ColorType
from pyPDP.utils.utils import unscale, get_selected_idx, convert_hyperparameters


@dataclass
class ICECurve:
    def __init__(self,
                 full_config_space: CS.ConfigurationSpace,
                 selected_hyperparameter: SelectedHyperparameterType,
                 x_ice: np.ndarray,
                 y_ice: np.ndarray,
                 y_variances: np.ndarray,
                 name="ICE-Curve"):
        """
        :param x_ice: Shape: (num_gridpoints, num_features) - scaled between [0..1]
        :param y_ice: Shape: (num_gridpoints)
        :param y_variances: (num_gridpoints)
        """
        self.full_config_space = full_config_space
        self.selected_hyperparameter = convert_hyperparameters(selected_hyperparameter, self.full_config_space)
        self.x_ice: np.ndarray = x_ice
        self.y_ice: np.ndarray = y_ice
        self.y_variances: np.ndarray = y_variances
        self.name = name

    def plot_values(self,
                    color="red",
                    ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)
        check_and_set_axis(ax, self.selected_hyperparameter)

        idx = get_selected_idx(self.selected_hyperparameter, self.full_config_space)
        x_unscaled = unscale(self.x_ice, self.full_config_space)

        # Switch cases for number of dimensions
        n_hyperparameters = len(self.selected_hyperparameter)
        if n_hyperparameters == 1:  # 1D
            x = x_unscaled[:, idx[0]]
            plot_line(x, self.y_ice, color=color, label=self.name, ax=ax)

        elif n_hyperparameters == 2:  # 2D
            x = x_unscaled[:, idx[0]]
            y = x_unscaled[:, idx[1]]
            plot_2D(x, y, self.y_ice, ax=ax)
        else:
            raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")

    def plot_incumbent(self,
                       color="white",
                       rounding=2,
                       ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)
        check_and_set_axis(ax, self.selected_hyperparameter)

        idx = get_selected_idx(self.selected_hyperparameter, self.full_config_space)
        x_unscaled = unscale(self.x_ice, self.full_config_space)

        # Switch cases for number of dimensions
        n_hyperparameters = len(self.selected_hyperparameter)
        min_idx = np.argmin(self.y_ice)
        min_val = self.y_ice[min_idx]
        if n_hyperparameters == 1:  # 1D
            x = x_unscaled[min_idx, idx[0]]
            ax.axvline(x, color=color)
            # TODO: Add annotation

        elif n_hyperparameters == 2:  # 2D
            x = x_unscaled[min_idx, idx[0]]
            y = x_unscaled[min_idx, idx[1]]
            ax.scatter(x, y, color=color, marker="*")
            ax.annotate(
                text=f"({x:.{rounding}f}, {y:.{rounding}f})",
                xy=(x, y),
                color=color,
                textcoords="offset points",
                xytext=(0, 10),  # distance from text to points (x,y)
                ha='center'
            )
        else:
            raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")

    def plot_confidences(self,
                         line_color="red",
                         gradient_color="xkcd:light red",
                         confidence_max_sigma: float = 1.5,
                         ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)
        check_and_set_axis(ax, self.selected_hyperparameter)

        idx = get_selected_idx(self.selected_hyperparameter, self.full_config_space)
        sigmas = np.sqrt(self.y_variances)
        x_unscaled = unscale(self.x_ice, self.full_config_space)

        # Switch cases for number of dimensions
        n_hyperparameters = len(self.selected_hyperparameter)
        if n_hyperparameters == 1:  # 1D
            x = x_unscaled[:, idx[0]]
            plot_1D_confidence_color_gradients(
                x=x,
                means=self.y_ice,
                stds=sigmas,
                max_sigma=confidence_max_sigma,
                color=gradient_color,
                ax=ax
            )
            plot_1D_confidence_lines(
                x=x,
                means=self.y_ice,
                stds=sigmas,
                k_sigmas=(1, 2),
                color=line_color,
                ax=ax,
                name=self.name
            )

        elif n_hyperparameters == 2:  # 2D
            x = x_unscaled[:, idx[0]]
            y = x_unscaled[:, idx[1]]
            plot_2D(x, y, sigmas, ax=ax)
        else:
            raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")

    @property
    def implied_config_space(self) -> CS.ConfigurationSpace:
        # Only works with Numerical Hyperparameter
        min_values = unscale(np.min(self.x_ice, axis=0), self.full_config_space)
        max_values = unscale(np.max(self.x_ice, axis=0), self.full_config_space)
        cs = CS.ConfigurationSpace()
        for hp, min_, max_ in zip(self.full_config_space.get_hyperparameters(), min_values, max_values):
            assert isinstance(hp, CSH.NumericalHyperparameter)
            if min_ == max_:
                hp_copy = CSH.Constant(hp.name, value=min_)
            else:
                hp_copy = hp.__class__(hp.name, lower=min_, upper=max_, log=hp.log)
            cs.add_hyperparameter(hp_copy)

        return cs


class ICE(Algorithm):
    """Individual Conditional Expectation"""

    def __init__(self,
                 surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 samples: np.ndarray,
                 num_grid_points_per_axis: int = 20,
                 seed=None):
        super().__init__(
            surrogate_model=surrogate_model,
            selected_hyperparameter=selected_hyperparameter,
            samples=samples,
            num_grid_points_per_axis=num_grid_points_per_axis,
            seed=seed
        )
        self.centered = False  # Can be set directly in class

        # Properties
        self._x_ice: Optional[np.ndarray] = None
        self._y_ice: Optional[np.ndarray] = None
        self._y_variances: Optional[np.ndarray] = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return ICECurve(
            full_config_space=self.config_space,
            selected_hyperparameter=self.selected_hyperparameter,
            x_ice=self.x_ice[idx],
            y_ice=self.y_ice[idx],
            y_variances=self.y_variances[idx],
            name=f"ICE-Curve[{idx}]"
        )

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
        cs = self.config_space
        idx = get_selected_idx(self.selected_hyperparameter, cs)
        num_features = len(cs.get_hyperparameters())

        # retrieve x-values from config
        x_s = self.grid_points

        # create x values by repeating x_s along a new dimension
        x_ice = self.samples.repeat(self.num_grid_points)
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

    @cached_property
    def grid_points(self) -> np.ndarray:
        single_axis_grid = np.linspace(0, 1, self.num_grid_points_per_axis)
        grid_axes = np.meshgrid(*[single_axis_grid for _ in range(self.n_selected_hyperparameter)])
        grid = np.stack(grid_axes).reshape((self.n_selected_hyperparameter, -1)).T
        return grid

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
            raise NotImplementedError("2D ICE Plot not available (Please use pdp instead)")
        else:
            raise NotImplementedError(f"Plotting for {self.n_selected_hyperparameter} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")
