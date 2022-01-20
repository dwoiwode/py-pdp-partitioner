import colorsys
import logging
import matplotlib.colors as mc

from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, List, Iterable, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt

from src.utils.utils import get_uniform_distributed_ranges, get_stds
from src.utils.typing import ColorType


class Plottable(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def plot(self,
             *args,
             x_hyperparameters: Optional[Iterable[CSH.Hyperparameter]] = None,
             ax: Optional[plt.Axes] = None):
        # x_hyperparameters = get_hyperparameters(x_hyperparameters, self.config_space)
        if x_hyperparameters is None:
            raise ValueError("x_hyperparameters is None and no config_space in class")
        n_hyperparameters = len(tuple(x_hyperparameters))
        if n_hyperparameters == 1:  # 1D
            raise NotImplemented("1D currently not implemented (#TODO)")
        elif n_hyperparameters == 2:  # 2D
            raise NotImplemented("2D currently not implemented (#TODO)")
        else:
            raise NotImplemented("Plotting for more than 2 dimensions not implemented. "
                                 "Please select a specific hp by setting `x_hyperparemeters`")


def adjust_lightness(color: ColorType,
                     amount=0.5) -> Tuple[float, float, float]:
    """
    From https://stackoverflow.com/a/49601444/7207309
    Adjust lightness for a given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> adjust_lightness('g', 0.3)
    >> adjust_lightness('#F034A3', 0.6)
    >> adjust_lightness((.3,.55,.1), 1.5)

    :param color: name, hex or rgb (scaled between 0..1)
    :param amount: amount of lightness change
        = 1 -> Equal
        < 1 -> Lighter
        > 1 -> Darker
    :return: New color as rgb-tuple scaled between 0..1
    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(h, max(0., min(1., amount * l)), s)


def get_color(color: ColorType) -> Tuple[float, float, float]:
    """
    :return: color as rgb-tuple scaled between 0..1
    """
    return mc.to_rgb(color)


def plot_function(f: Callable[[Any], float],
                  cs: CS.ConfigurationSpace,
                  samples_per_axis=100,
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = get_ax(ax)
    parameters = cs.get_hyperparameters()
    n_parameter = len(parameters)
    if n_parameter > 2:
        raise ValueError("Plotting currently only supports max 2 dimensions")
    if n_parameter == 0:
        raise ValueError("Requires at least 1 parameter for plotting")

    ranges = get_uniform_distributed_ranges(cs, samples_per_axis=samples_per_axis)

    x = ranges[0]
    ax.set_xlabel(parameters[0].name)
    if n_parameter == 1:
        # plot ground truth lines
        y = [f(p) for p in x]
        ax.plot(x, y, label=f.__name__, c='black')
    elif n_parameter == 2:
        # plot ground truth lines
        y = ranges[1]
        xx, yy = np.meshgrid(x, y)
        z = []
        for x1, x2 in zip(xx.reshape(-1), yy.reshape(-1)):
            X = {parameters[0].name: x1, parameters[1].name: x2}
            z.append(f(**X))

        z = np.reshape(z, xx.shape)
        plt.pcolormesh(x, y, z, shading='auto')

    return ax


def plot_line(x: np.ndarray,
              y: np.ndarray,
              color: ColorType = "black",
              label:Optional[str] = None,
              ax: Optional[plt.Axes] = None):
    ax = get_ax(ax)
    ax.plot(x, y, color=color, label=label)


# def plot_model_confidence(optimizer: Sampler, cs: CS.ConfigurationSpace,
#                           samples_per_axis=100, ax: Optional[plt.Axes] = None) -> plt.Axes:
#     ax = get_ax(ax)
#
#     if len(cs.get_hyperparameters()) != 1:
#         raise ValueError("Number of hyperparameters for plotting model confidence has to be 1")
#
#     ranges = get_uniform_distributed_ranges(cs, samples_per_axis, scaled=False)
#     scaled_ranges = get_uniform_distributed_ranges(cs, samples_per_axis, scaled=True)
#     mu, std = optimizer.surrogate_score([Configuration(cs, vector=vector) for vector in np.asarray(scaled_ranges).T])
#
#     plot_confidence_lists(ranges[0], mu, std, ax=ax)
#     return ax


# def plot_acquisition(acquisition_function: AcquisitionFunction, cs: CS.ConfigurationSpace,
#                      samples_per_axis=100, ax: Optional[plt.Axes] = None) -> plt.Axes:
#     if len(cs.get_hyperparameters()) != 1:
#         raise ValueError("Number of hyperparameters for plotting model confidence has to be 1")
#
#     ax = get_ax(ax)
#
#     # ranges = _get_uniform_distributed_ranges(cs, samples_per_axis, scaled=False)[0]
#     # scaled_ranges = _get_uniform_distributed_ranges(cs, samples_per_axis, scaled=True)
#     configs = cs.sample_configuration(samples_per_axis)
#     acquisition_y = np.asarray([acquisition_function(x) for x in configs]).reshape(-1)
#     ranges = np.asarray([list(config.values())[0] for config in configs])
#
#     order = np.argsort(ranges)
#     ranges = ranges[order]
#     acquisition_y = acquisition_y[order]
#
#     ax.fill_between(ranges, acquisition_y, color="darkgreen", alpha=0.3)
#     ax.plot(ranges, acquisition_y, color="darkgreen", label=acquisition_function.__class__.__name__)
#
#     config = acquisition_function.get_optimum()
#     ax.plot(list(config.values())[0], acquisition_function(config), "*", color="red", label="next best candidate",
#             markersize=15)
#
#     return ax


def plot_1D_confidence_color_gradients(x: np.ndarray,
                                       means: np.ndarray,
                                       stds: Optional[np.ndarray] = None, variances: Optional[np.ndarray] = None,
                                       color: ColorType = "lightblue",
                                       sigma_steps: int = 100,
                                       max_sigma: float = 1.5,
                                       ax: Optional[plt.Axes] = None):
    ax = get_ax(ax)
    color = get_color(color)
    stds = get_stds(stds, variances)

    for i in range(sigma_steps):
        sigma_factor = i / sigma_steps * max_sigma
        ax.fill_between(x,
                        y1=means - sigma_factor * stds,
                        y2=means + sigma_factor * stds,
                        alpha=0.3 / sigma_steps, color=color)


def plot_1D_confidence_lines(x: np.ndarray,
                             means: np.ndarray,
                             stds: Optional[np.ndarray] = None, variances: Optional[np.ndarray] = None,
                             color: ColorType = "lightblue",
                             k_sigmas: Iterable[float] = (0, 1, 2),
                             name: str = "model",
                             ax: Optional[plt.Axes] = None):
    stds = get_stds(stds, variances)
    ax = get_ax(ax)

    # Handle zero first
    if 0 in k_sigmas:
        ax.plot(x, means, color=color, alpha=0.3, label=f"{name}-$\mu$")

    for k_sigma in sorted(k_sigmas):  # Sort for order in labels
        if k_sigma == 0:
            continue
        # If int omit decimal points, if float: print 2 decimal points
        if isinstance(k_sigma, int):
            label = f"{name}-$\mu\pm${k_sigma}$\sigma$"
        else:
            label = f"{name}-$\mu\pm${k_sigma:.2f}$\sigma$"
        ax.plot(x, means - stds, color=color, alpha=1 / k_sigma * 0.2, label=label)
        ax.plot(x, means + stds, color=color, alpha=1 / k_sigma * 0.2)


def get_ax(ax: Optional[plt.Axes]) -> plt.Axes:
    if ax is None:
        fig = plt.gcf()
        assert isinstance(fig, plt.Figure)
        ax = fig.gca()

    assert isinstance(ax, plt.Axes)
    return ax
