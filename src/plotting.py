import warnings
from typing import Callable, Any, Optional, List

import ConfigSpace as CS
import numpy as np
from ConfigSpace import Configuration, hyperparameters as CSH
from matplotlib import pyplot as plt

from src.optimizer import AcquisitionFunction


def _get_ax(ax: Optional[plt.Axes]) -> plt.Axes:
    if ax is None:
        fig = plt.gcf()
        assert isinstance(fig, plt.Figure)
        ax = fig.gca()

    assert isinstance(ax, plt.Axes)
    return ax


def _get_uniform_distributed_ranges(cs: CS.ConfigurationSpace, samples_per_axis: int = 100, scaled=False) -> List[np.ndarray]:
    ranges = []
    for parameter in cs.get_hyperparameters():
        assert isinstance(parameter, CSH.NumericalHyperparameter)
        if scaled:
            ranges.append(np.linspace(0, 1, num=samples_per_axis))
        else:
            if parameter.log:
                space_function = np.logspace
            else:
                space_function = np.linspace

            ranges.append(space_function(parameter.lower, parameter.upper, num=samples_per_axis))
    return ranges


def plot_function(f: Callable[[Any], float], cs: CS.ConfigurationSpace,
                  samples_per_axis=100, ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = _get_ax(ax)
    parameters = cs.get_hyperparameters()
    n_parameter = len(parameters)
    if n_parameter > 2:
        raise ValueError("Plotting currently only supports max 2 dimensions")
    if n_parameter == 0:
        raise ValueError("Requires at least 1 parameter for plotting")

    ranges = _get_uniform_distributed_ranges(cs, samples_per_axis=samples_per_axis)

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


def plot_samples(configs: List[Configuration], y: List[float], ax=None, plotting_kwargs=None):
    # Basic usage checks
    if len(configs) != len(y):
        warnings.warn(f"Expect x and y to have the same length! ({len(configs)} != {len(y)})")

    # No data points -> Cannot plot something
    if len(configs) == 0:
        return

    if plotting_kwargs is None:
        plotting_kwargs = {"color": "red"}

    ax = _get_ax(ax)

    parameters = configs[0].values()
    n_parameter = len(parameters)
    if n_parameter == 0:
        raise ValueError("Requires at least 1 parameter for plotting")
    elif n_parameter == 1:
        x = [tuple(config.values())[0] for config in configs]
        ax.scatter(x, y, **plotting_kwargs)

    elif n_parameter == 2:
        # TODO(dwoiwode): How to plot y?
        x1, x2 = zip(*[list(config.values()) for config in configs])
        ax.scatter(x1, x2, **plotting_kwargs)
    if n_parameter > 2:
        raise ValueError("Plotting currently only supports max 2 dimensions")

    return ax


def plot_model_confidence(model: Callable[[Any], float], cs: CS.ConfigurationSpace,
                          samples_per_axis=100, ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = _get_ax(ax)

    if len(cs.get_hyperparameters()) != 1:
        raise ValueError("Number of hyperparameters for plotting model confidence has to be 1")

    ranges = _get_uniform_distributed_ranges(cs, samples_per_axis, scaled=False)
    scaled_ranges = _get_uniform_distributed_ranges(cs, samples_per_axis, scaled=True)
    mu, std = model(scaled_ranges[0])

    sigma_steps = 100
    max_sigma = 1.5
    x = ranges[0]
    for i in range(sigma_steps):
        sigma_factor = i / sigma_steps * max_sigma
        ax.fill_between(x,
                        y1=mu - sigma_factor * std,
                        y2=mu + sigma_factor * std,
                        alpha=0.3 / sigma_steps, color="lightblue")

    ax.plot(x, mu, color="blue", alpha=0.3, label="GP-$\mu$")
    ax.plot(x, mu - std, color="blue", alpha=0.2, label="GP-1$\sigma$")
    ax.plot(x, mu + std, color="blue", alpha=0.2)
    ax.plot(x, mu - 2 * std, color="blue", alpha=0.1, label="GP-2$\sigma$")
    ax.plot(x, mu + 2 * std, color="blue", alpha=0.1)
    return ax

def plot_acquisition(acquisition_function: AcquisitionFunction, cs: CS.ConfigurationSpace,
                          samples_per_axis=100, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if len(cs.get_hyperparameters()) != 1:
        raise ValueError("Number of hyperparameters for plotting model confidence has to be 1")

    ax = _get_ax(ax)

    # ranges = _get_uniform_distributed_ranges(cs, samples_per_axis, scaled=False)[0]
    # scaled_ranges = _get_uniform_distributed_ranges(cs, samples_per_axis, scaled=True)
    configs = cs.sample_configuration(samples_per_axis)
    acquisition_y = np.asarray([acquisition_function(x) for x in configs]).reshape(-1)
    ranges = np.asarray([list(config.values())[0] for config in configs])

    order = np.argsort(ranges)
    ranges = ranges[order]
    acquisition_y = acquisition_y[order]

    ax.fill_between(ranges, acquisition_y, color="darkgreen", alpha=0.3)
    ax.plot(ranges, acquisition_y, color="darkgreen", label=acquisition_function.__class__.__name__)

    config = acquisition_function.get_optimum()
    ax.plot(list(config.values())[0], acquisition_function(config), "*", color="red", label="next best candidate", markersize=15)

    return ax


def style_axes(ax: plt.Axes, cs: CS.ConfigurationSpace):
    parameters = cs.get_hyperparameters()
    n_parameters = len(parameters)

    if n_parameters == 1:
        ax.set_xlabel(f"{parameters[0].name}")
        ax.set_ylabel(f"f({parameters[0].name})")
    elif n_parameters == 2:
        plt.axis('scaled')
        ax.set_xlabel(f"{parameters[0].name}")
        ax.set_ylabel(f"{parameters[1].name}")


def finalize_figure(fig: plt.Figure):
    fig.legend()
    fig.tight_layout()
