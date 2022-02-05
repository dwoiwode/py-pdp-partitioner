from matplotlib import pyplot as plt

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

    bo.surrogate_model.plot(ax=ax_function)
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
    pdp.plot(ax=ax, line_color="black", gradient_color="blue", with_confidence=True)
    plt.legend()
    plt.tight_layout()
    return fig