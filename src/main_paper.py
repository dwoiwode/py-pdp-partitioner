from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
from tqdm import tqdm

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.algorithms.pdp import PDP
from src.blackbox_functions import BlackboxFunction
from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler.acquisition_function import LowerConfidenceBound
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.surrogate_models import GaussianProcessSurrogate

seed = 0


# def plot_full_bayesian(bo: BayesianOptimizationSampler) -> plt.Figure:
#     fig = plt.figure(figsize=(16, 9))
#     assert isinstance(fig, plt.Figure)
#     ax_function = fig.add_subplot(3, 1, (1, 2))
#     ax_acquisition = fig.add_subplot(3, 1, 3)
#     assert isinstance(ax_function, plt.Axes)
#     assert isinstance(ax_acquisition, plt.Axes)
#
#     bo.surrogate_model.plot(ax=ax_function)
#     bo.plot(ax=ax_function, x_hyperparameters="x1")
#     bo.acq_func.plot(ax=ax_acquisition, x_hyperparameters="x1")
#     plt.legend()
#     plt.tight_layout()
#     return fig


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


def run_algorithm(f, cs, bo_samples, n_splits=1, tau=0.1):
    current_run_name = f"{bo_samples}_{f.__name__}"
    selected_hyperparameter = cs.get_hyperparameter("x1")

    # Sampler
    acq_class = LowerConfidenceBound
    acq_class_kwargs = {'tau': tau}
    bo_sampler = BayesianOptimizationSampler(f, cs, initial_points=4*f.ndim,
                                             seed=seed, acq_class=acq_class, acq_class_kwargs=acq_class_kwargs)
    bo_sampler.sample(bo_samples)

    # Surrogate model
    surrogate_model = GaussianProcessSurrogate(cs)
    surrogate_model.fit(bo_sampler.X, bo_sampler.y)

    # ICE
    ice = ICE(surrogate_model, selected_hyperparameter)

    # PDP
    pdp = PDP.from_ICE(ice)

    # Partitioner
    dt_partitioner = DTPartitioner.from_ICE(ice)
    dt_partitioner.partition(n_splits)
    best_region = dt_partitioner.get_incumbent_region(bo_sampler.incumbent[0])

    # metrics
    mc_root = dt_partitioner.root.region.mean_confidence
    nll_root = dt_partitioner.root.region.negative_log_likelihood(f)
    nll = best_region.negative_log_likelihood(f)
    mc = best_region.mean_confidence
    delta_mc = mc / mc_root
    delta_nll = nll / nll_root

    mmd = bo_sampler.maximum_mean_discrepancy(1000)

    print(f'f: {f}, n_splits: {n_splits}, sampling_points: {sampling_points}')
    print(f'mmd: {mmd}')
    print(f'mc: {mc}, mc_root: {mc_root}, delta_mc: {delta_mc=}')
    print(f'nll: {nll}, {nll_root=}, {delta_nll=}')

    # Plots
    # folder = Path("../plots")
    # folder.mkdir(parents=True, exist_ok=True)
    #
    # fig_pdp = plot_full_pdp(pdp)
    # fig_pdp.savefig(folder / f"{current_run_name}_pdp.jpg")




if __name__ == '__main__':
    functions:List[BlackboxFunction] = [
        StyblinskiTang.for_n_dimensions(3),
        # StyblinskiTang.for_n_dimensions(5),
        # StyblinskiTang.for_n_dimensions(8),
    ]
    bo_sampling_points = [80, 150, 250]

    for f, sampling_points in zip(functions, bo_sampling_points):
        # for sampling_points in tqdm(bo_sampling_points, desc="Bayesian sampling points"):
        run_algorithm(f, f.config_space, sampling_points, n_splits=1)
