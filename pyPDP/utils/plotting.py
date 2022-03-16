import colorsys
from typing import Callable, Any, Optional, List, Iterable, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import matplotlib.colors as mc
import numpy as np
from matplotlib import pyplot as plt

from pyPDP.utils.typing import ColorType, SelectedHyperparameterType
from pyPDP.utils.utils import get_uniform_distributed_ranges, get_stds, get_hyperparameters


# Resolve/Getter functions (that resolve function inputs)
def get_color(color: ColorType) -> Tuple[float, float, float]:
    """
    Input can be either string (name or hex) or rgb-tuple (scale between 0..1).
    Output is always rgb-tuple scaled between 0..1
    """
    return mc.to_rgb(color)


def get_random_color() -> ColorType:
    rgb = np.random.uniform(size=3)
    return tuple(rgb)


def get_ax(ax: Optional[plt.Axes]) -> plt.Axes:
    """
    Return input axis. If it is None, return current axis instead (plt.gca())
    """
    if ax is None:
        fig = plt.gcf()
        assert isinstance(fig, plt.Figure)
        ax = fig.gca()

    assert isinstance(ax, plt.Axes)
    return ax


def check_and_set_axis(ax: plt.Axes, hyperparameters: List[CSH.Hyperparameter], set_bounds=True, ylabel="Prediction"):
    """
    Set axis labels and types (e.g. log) for all selected hyperparameters.
    If this does not fit with current configuration raise Error instead
    """
    assert isinstance(ax, plt.Axes)

    n_hyperparameters = len(hyperparameters)
    if n_hyperparameters == 1:
        hp = hyperparameters[0]

        # Check x label
        current_label = ax.get_xlabel()
        new_label = hp.name
        if current_label != "" and current_label != new_label:
            raise ValueError(f"Current x label is {current_label}, but tried to set label to {new_label}. "
                             f"Did you mess up the plots?")

        # Check y label
        current_label = ax.get_ylabel()
        if current_label != "" and current_label != ylabel:
            raise ValueError(f"Current y label is {current_label}, but tried to set label to {ylabel}. "
                             f"Did you mess up the plots?")

        # Set Label
        ax.set_xlabel(new_label)
        ax.set_ylabel(ylabel)

        # Set Bounds 1D
        if set_bounds and isinstance(hp, CSH.NumericalHyperparameter):
            if hp.log:
                ax.set_xscale("log")
            ax.set_xlim(hp.lower, hp.upper)
    elif n_hyperparameters == 2:
        hp1, hp2 = hyperparameters

        # Check Label
        current_label = ax.get_xlabel(), ax.get_ylabel()
        new_label = hp1.name, hp2.name
        if current_label != ("", "") and current_label != new_label:
            raise ValueError(f"Current label is {current_label}, but tried to set label to {new_label}. "
                             f"Did you mess up the plots?")
        # Set Label
        ax.set_xlabel(new_label[0])
        ax.set_ylabel(new_label[1])

        # Numerical axis
        # Set Bounds 2D
        if set_bounds and isinstance(hp1, CSH.NumericalHyperparameter):
            if hp1.log:
                ax.set_xscale("log")
            ax.set_xlim(hp1.lower, hp1.upper)
        if set_bounds and isinstance(hp2, CSH.NumericalHyperparameter):
            if hp2.log:
                ax.set_yscale("log")
            ax.set_ylim(hp2.lower, hp2.upper)


# Utils
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


# Plotting helper
def plot_function(f: Callable[[Any], float],
                  cs: CS.ConfigurationSpace,
                  samples_per_axis=100,
                  color: ColorType = "black",
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = get_ax(ax)
    color = get_color(color)

    hps = cs.get_hyperparameters()
    constants = {hp.name: hp.value for hp in hps if isinstance(hp, CSH.Constant)}
    parameters = [hp for hp in hps if not isinstance(hp, CSH.Constant)]
    n_parameter = len(parameters)
    if n_parameter > 2:
        raise ValueError("Plotting currently only supports max 2 dimensions")
    if n_parameter == 0:
        raise ValueError("Requires at least 1 parameter for plotting")

    ranges = get_uniform_distributed_ranges(parameters, samples_per_axis=samples_per_axis)

    x = ranges[0]
    check_and_set_axis(ax, parameters)
    if n_parameter == 1:
        # plot ground truth lines
        y = [f(**{parameters[0].name: p}, **constants) for p in x]
        ax.plot(x, y, label=f.__name__, color=color)
    elif n_parameter == 2:
        # plot ground truth lines
        y = ranges[1]
        xx, yy = np.meshgrid(x, y)
        z = []
        for x1, x2 in zip(xx.reshape(-1), yy.reshape(-1)):
            X = {parameters[0].name: x1, parameters[1].name: x2}
            z.append(f(**X, **constants))

        z = np.reshape(z, xx.shape)
        ax.pcolormesh(x, y, z, shading='auto')

    return ax


def plot_line(
        x: np.ndarray,
        y: np.ndarray,
        color: ColorType = "black",
        label: Optional[str] = None,
        ax: Optional[plt.Axes] = None
):
    ax = get_ax(ax)
    ax.plot(x, y, color=color, label=label)


def plot_1D_confidence_color_gradients(
        x: np.ndarray,
        means: np.ndarray,
        stds: Optional[np.ndarray] = None,  # Choose stds or variances
        variances: Optional[np.ndarray] = None,  # Choose stds or variances
        color: ColorType = "lightblue",
        sigma_steps: int = 100,
        max_sigma: float = 1.5,
        ax: Optional[plt.Axes] = None
):
    ax = get_ax(ax)
    color = get_color(color)
    stds = get_stds(stds, variances)

    for i in range(sigma_steps):
        sigma_factor = i / sigma_steps * max_sigma
        ax.fill_between(
            x,
            y1=means - sigma_factor * stds,
            y2=means + sigma_factor * stds,
            alpha=0.3 / sigma_steps,
            color=color
        )


def plot_1D_confidence_lines(
        x: np.ndarray,
        means: np.ndarray,
        stds: Optional[np.ndarray] = None,  # Choose stds or variances
        variances: Optional[np.ndarray] = None,  # Choose stds or variances
        color: ColorType = "lightblue",
        k_sigmas: Iterable[float] = (0, 1, 2),
        name: str = "model",
        ax: Optional[plt.Axes] = None
):
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
        ax.plot(x, means - k_sigma * stds, color=color, alpha=1 / k_sigma * 0.2, label=label)
        ax.plot(x, means + k_sigma * stds, color=color, alpha=1 / k_sigma * 0.2)


def plot_2D(
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        ax: Optional[plt.Axes] = None
):
    ax = get_ax(ax)

    # Colors
    ax.tripcolor(x, y, values)

    # Contours/Labels
    contour = ax.tricontour(x, y, values, colors="black")
    contour.clabel(contour.levels, fontsize=12, colors="black", inline=True)


def plot_config_space(config_space: CS.ConfigurationSpace,
                      x_hyperparameters: Optional[SelectedHyperparameterType] = None,
                      color: ColorType = "orange",
                      alpha: float = 0.5,
                      ax: Optional[plt.Axes] = None):
    """
    Draws a box around (selected) hyperparameter bounds of a config_space
    """
    ax = get_ax(ax)
    color = get_color(color)
    x_hyperparameters = get_hyperparameters(x_hyperparameters, config_space)
    check_and_set_axis(ax, x_hyperparameters, set_bounds=False)

    n_hyperparameters = len(x_hyperparameters)
    if n_hyperparameters == 1:
        # Plot 1D
        hp = x_hyperparameters[0]
        assert isinstance(hp, CSH.NumericalHyperparameter)
        ax.axvline(hp.lower, color=color)
        ax.axvline(hp.upper, color=color)
        ax.axvspan(hp.lower, hp.upper, alpha=alpha, color=color)
    elif n_hyperparameters == 2:
        # Plot 2D
        x1, x2 = x_hyperparameters
        if isinstance(x1, CSH.NumericalHyperparameter):
            x_lower = x1.lower
            x_upper = x1.upper
        elif isinstance(x1, CSH.Constant):
            x_lower = x1.value
            x_upper = x1.value
            alpha = 1
        else:
            raise TypeError(f"{x1} currently not supported for plotting!")

        if isinstance(x2, CSH.NumericalHyperparameter):
            y_lower = x2.lower
            y_upper = x2.upper
        elif isinstance(x2, CSH.Constant):
            y_lower = x2.value
            y_upper = x2.value
            alpha = 1
        else:
            raise TypeError(f"{x2} currently not supported for plotting!")

        ax.fill_betweenx(
            y=[y_lower, y_upper],
            x1=x_lower, x2=x_upper,
            alpha=alpha, color=color
        )
    else:
        raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                  "Please select a specific hp by setting `selected_hyperparameters`")


def as_quadratic_shape_as_possible_for_n_figures(n: int) -> Tuple[int, int]:
    w = int(np.ceil(np.sqrt(n)))
    for h in range(1, w + 1):
        if h * w >= n:
            return w, h
    raise ValueError(f"Could not find valid w, h for {n} ({w=})")
