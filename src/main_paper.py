from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
from tqdm import tqdm

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.algorithms.pdp import PDP
from src.blackbox_functions import synthetic_functions, BlackboxFunction, config_space_nd
from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.surrogate_models import GaussianProcessSurrogate

seed = 0


def plot_full_bayesian(bo: BayesianOptimizationSampler) -> plt.Figure:
    fig = plt.figure(figsize=(16, 9))
    assert isinstance(fig, plt.Figure)
    ax_function = fig.add_subplot(3, 1, (1, 2))
    ax_acquisition = fig.add_subplot(3, 1, 3)
    assert isinstance(ax_function, plt.Axes)
    assert isinstance(ax_acquisition, plt.Axes)

    bo.surrogate_model.plot(ax=ax_function)
    bo.plot(ax=ax_function, x_hyperparameters="x1")
    bo.acq_func.plot(ax=ax_acquisition, x_hyperparameters="x1")
    plt.legend()
    plt.tight_layout()
    return fig


def plot_full_pdp(pdp: PDP) -> plt.Figure:
    fig = plt.figure(figsize=(16, 9))
    assert isinstance(fig, plt.Figure)
    ax = fig.gca()
    assert isinstance(ax, plt.Axes)

    pdp.ice.plot(ax=ax)
    pdp.plot(ax=ax, line_color="black", gradient_color="blue", with_confidence=True)
    plt.legend()
    plt.tight_layout()
    return fig


def run_algorithm(f, cs, bo_samples):
    current_run_name = f"{bo_samples}_{f.__name__}"
    selected_hyperparameter = cs.get_hyperparameter("x1")

    # Sampler
    bo_sampler = BayesianOptimizationSampler(f, cs, seed=seed)
    bo_sampler.sample(bo_samples)

    # Surrogate model
    surrogate_model = GaussianProcessSurrogate(cs)
    surrogate_model.fit(bo_sampler.X, bo_sampler.y)

    # ICE
    ice = ICE(surrogate_model, selected_hyperparameter)

    # PDP
    pdp = PDP.from_ICE(ice)

    # Partitioner
    # dt_partitioner = DTPartitioner.from_ICE(ice)

    # Plots
    folder = Path("../plots")
    folder.mkdir(parents=True, exist_ok=True)

    fig_bo = plot_full_bayesian(bo_sampler)
    fig_bo.savefig(folder / f"{current_run_name}_bayesian_optimization.jpg")

    fig_pdp = plot_full_pdp(pdp)
    fig_pdp.savefig(folder / f"{current_run_name}_pdp.jpg")


if __name__ == '__main__':
    functions:List[BlackboxFunction] = [
        StyblinskiTang(3),
        StyblinskiTang(5),
        StyblinskiTang(8),
    ]
    bo_sampling_points = [80, 150, 250]

    for f in tqdm(functions, desc="Function"):
        for sampling_points in tqdm(bo_sampling_points, desc="Bayesian sampling points"):
            run_algorithm(f, f.config_space, sampling_points)
