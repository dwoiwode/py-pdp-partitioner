from typing import Optional, Tuple

from matplotlib import pyplot as plt

from src.algorithms.ice import ICECurve
from src.algorithms.pdp import PDP
from src.sampler.bayesian_optimization import BayesianOptimizationSampler


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
