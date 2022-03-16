from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from pyPDP.algorithms.ice import ICECurve, ICE
from pyPDP.algorithms.pdp import PDP
from pyPDP.sampler.bayesian_optimization import BayesianOptimizationSampler
from pyPDP.surrogate_models import SurrogateModel
from pyPDP.utils.plotting import as_quadratic_shape_as_possible_for_n_figures


def plot_full_bayesian(bo: BayesianOptimizationSampler) -> plt.Figure:
    """
    Plots Samples, Surrogate model used by Bayesian Optimizer (with confidence) and Acquisition Function
    :param bo:
    :return:
    """
    fig = plt.figure(figsize=(16, 9))
    assert isinstance(fig, plt.Figure)
    ax_function = fig.add_subplot(3, 1, (1, 2))
    ax_acquisition = fig.add_subplot(3, 1, 3)
    assert isinstance(ax_function, plt.Axes)
    assert isinstance(ax_acquisition, plt.Axes)

    bo.surrogate_model.plot_means(ax=ax_function)
    bo.surrogate_model.plot_confidences(ax=ax_function)
    bo.plot(ax=ax_function, x_hyperparameters="x1")
    bo.acq_func.plot(ax=ax_acquisition, x_hyperparameters="x1")
    ax_acquisition.plot(bo.acq_func.get_optimum()["x1"], marker="x")
    plt.legend()
    plt.tight_layout()
    return fig


def plot_full_pdp(pdp: PDP) -> plt.Figure:
    """
    Plots all ice-curves + pdp-curve with confidence
    :param pdp:
    :return:
    """
    fig = plt.figure(figsize=(16, 9))
    assert isinstance(fig, plt.Figure)
    ax = fig.gca()
    assert isinstance(ax, plt.Axes)

    pdp.ice.plot(ax=ax)
    pdp.plot_values(ax=ax, color="black")
    pdp.plot_confidences(ax=ax, line_color="black", gradient_color="blue")
    plt.legend()
    plt.tight_layout()
    return fig


def plot_2D_ICE_Curve_with_confidence(
        ice_curve: ICECurve,
        *,
        fig: Optional[plt.Figure] = None
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    # Create figure/axes
    if fig is None:
        fig = plt.figure(figsize=(16, 9))
    assert isinstance(fig, plt.Figure)

    ax_mean = fig.add_subplot(1, 2, 1)
    ax_sigma = fig.add_subplot(1, 2, 2)
    assert isinstance(ax_mean, plt.Axes)
    assert isinstance(ax_sigma, plt.Axes)

    # Plotting
    ice_curve.plot_values(ax=ax_mean)
    ice_curve.plot_incumbent(ax=ax_mean)
    ice_curve.plot_confidences(ax=ax_sigma)
    ice_curve.plot_incumbent(ax=ax_sigma)

    # Styling
    name = ice_curve.name
    ax_mean.set_title(f"{name} Means")
    ax_sigma.set_title(f"{name} Confidences")
    return fig, (ax_mean, ax_sigma)


def plot_hyperparameter_array_1D(
        surrogate_model: SurrogateModel, *,
        num_samples=1000,
        num_grid_points_per_axis=20,
        fig_res=4,
        seed=0
) -> plt.Figure:
    cs = surrogate_model.config_space
    hyperparameters = cs.get_hyperparameters()
    w, h = as_quadratic_shape_as_possible_for_n_figures(len(hyperparameters))

    fig, axs = plt.subplots(h, w, sharey="all", figsize=(w * fig_res, h * fig_res))
    for selected_hp, ax in zip(hyperparameters, np.reshape(axs, -1)):
        ice = ICE.from_random_points(
            surrogate_model,
            selected_hp,
            num_samples=num_samples,
            num_grid_points_per_axis=num_grid_points_per_axis
        )
        pdp = PDP.from_ICE(ice, seed)
        pdp.plot_values(ax=ax)
        pdp.plot_confidences(ax=ax)
        ax.legend()
    return fig


def plot_hyperparameter_array_2D(
        surrogate_model: SurrogateModel, *,
        num_samples=1000,
        num_grid_points_per_axis=20,
        fig_res=4,
        seed=0
) -> Tuple[plt.Figure, plt.Figure]:
    cs = surrogate_model.config_space
    hyperparameters = cs.get_hyperparameters()
    n = len(hyperparameters)
    fig_mean, axs_mean = plt.subplots(n, n, figsize=(n * fig_res, n * fig_res))
    fig_std, axs_std = plt.subplots(n, n, figsize=(n * fig_res, n * fig_res))
    if n == 1:
        # In this case plt.subplots returns only a "plt.AxesSubplot"
        axs_std = [[axs_std]]
        axs_mean = [[axs_mean]]
    for selected_hp2, ax_mean_row, ax_std_row in zip(hyperparameters, axs_mean, axs_std):
        for selected_hp1, ax_mean, ax_std in zip(hyperparameters, ax_mean_row, ax_std_row):
            if selected_hp1 == selected_hp2:
                hps = (selected_hp1,)
            else:
                hps = (selected_hp1, selected_hp2)
            pdp = PDP.from_random_points(
                surrogate_model,
                selected_hyperparameter=hps,
                seed=seed,
                num_samples=num_samples,
                num_grid_points_per_axis=num_grid_points_per_axis
            )
            if len(hps) == 1:
                pdp.plot_values(ax=ax_mean)
                pdp.plot_confidences(ax=ax_mean)
                pdp.plot_confidences(ax=ax_std)
                ax_mean.legend()
                ax_std.legend()
            else:
                pdp.plot_values(ax=ax_mean)
                pdp.plot_confidences(ax=ax_std)

    return fig_mean, fig_std
