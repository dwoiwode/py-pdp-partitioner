import warnings
from typing import List, Dict, Iterable, Type

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.algorithms.pdp import PDP
from src.blackbox_functions import BlackboxFunction, BlackboxFunctionND
from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler import Sampler
from src.sampler.acquisition_function import LowerConfidenceBound
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.sampler.random_sampler import RandomSampler
from src.surrogate_models import GaussianProcessSurrogate
from src.utils.plotting import plot_function, plot_config_space

warnings.filterwarnings("ignore", category=ConvergenceWarning)

seed = 0


def run_algorithm(f: BlackboxFunction, bo_samples: int, n_splits: int = 1, tau: float = 0.1):
    current_run_name = f"{bo_samples}_{f.__name__}"
    cs = f.config_space
    selected_hyperparameter = cs.get_hyperparameter("x1")

    # Sampler
    acq_class = LowerConfidenceBound
    acq_class_kwargs = {'tau': tau}
    bo_sampler = BayesianOptimizationSampler(f, cs, initial_points=4 * f.ndim,
                                             seed=seed, acq_class=acq_class, acq_class_kwargs=acq_class_kwargs)
    bo_sampler.sample(bo_samples)

    # Surrogate model
    surrogate_model = GaussianProcessSurrogate(cs)
    surrogate_model.fit(bo_sampler.X, bo_sampler.y)

    # ICE
    ice = ICE.from_random_points(surrogate_model, selected_hyperparameter)

    # PDP
    pdp = PDP.from_ICE(ice)

    # Partitioner
    dt_partitioner = DTPartitioner.from_ICE(ice)
    dt_partitioner.partition(n_splits)
    best_region = dt_partitioner.get_incumbent_region(bo_sampler.incumbent[0])

    # metrics
    mc_root = dt_partitioner.root.mean_confidence
    nll_root = dt_partitioner.root.negative_log_likelihood(f)
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


def figure_1_3(f: BlackboxFunction = StyblinskiTang.for_n_dimensions(2, seed=0),
               samplers: Dict[str, Sampler] = None,
               sampled_points=50):
    cs = f.config_space
    if samplers is None:
        # Default sampler (from paper)
        samplers = {
            "High sampling bias": BayesianOptimizationSampler(
                f,
                cs,
                acq_class=LowerConfidenceBound,
                acq_class_kwargs={"tau": 0.1}
            ),
            "Medium Sampling bias": BayesianOptimizationSampler(
                f,
                cs,
                acq_class=LowerConfidenceBound,
                acq_class_kwargs={"tau": 2}
            ),
            "Low Sampling bias": RandomSampler(
                f,
                cs,
            )
        }

    f_pd = f.pd_integral("x2")

    n = len(samplers)
    fig1, axes1 = plt.subplots(1, n, sharex=True, sharey=True, figsize=(4 * n, 4))
    fig3, axes3 = plt.subplots(1, n, sharex=True, sharey=True, figsize=(4 * n, 4))
    for (name, sampler), ax1, ax3 in zip(samplers.items(), axes1, axes3):
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax3, plt.Axes)
        assert isinstance(sampler, Sampler)
        sampler.sample(sampled_points)
        surrogate = GaussianProcessSurrogate(cs, seed=seed)
        surrogate.fit(sampler.X, sampler.y)
        pdp = PDP.from_random_points(surrogate_model=surrogate, selected_hyperparameter="x1")

        # Figure 1
        plot_function(f, cs, samples_per_axis=200, ax=ax1)
        sampler.plot("red", ax=ax1)
        ax1.set_title(name)

        # Figure 3
        plot_function(f_pd, f_pd.config_space, samples_per_axis=200, ax=ax3)
        pdp.plot(line_color="blue", gradient_color="lightblue", with_confidence=True, ax=ax3)
        ax3.set_title(name)

    fig1.savefig("Figure 1.png")
    fig3.savefig("Figure 3.png")
    plt.show()


def figure_2(f: BlackboxFunction = StyblinskiTang.for_n_dimensions(2, seed=seed)):
    cs = f.config_space
    sampler = BayesianOptimizationSampler(
        f,
        cs,
        acq_class=LowerConfidenceBound,
        acq_class_kwargs={"tau": 0.1}
    )

    # Figure/Axes
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, figure=fig)
    ax_2D_plot = fig.add_subplot(gs[:, 0])
    ax_selected_1 = fig.add_subplot(gs[0, 1])
    ax_selected_2 = fig.add_subplot(gs[1, 1])
    assert isinstance(ax_2D_plot, plt.Axes)
    assert isinstance(ax_selected_1, plt.Axes)
    assert isinstance(ax_selected_2, plt.Axes)
    ax_selected_1.set_ylim(-75, 150)
    ax_selected_2.set_ylim(-75, 150)

    # 2D Plot
    plot_function(f, cs, samples_per_axis=200, ax=ax_2D_plot)
    sampler.sample(40)
    sampler.plot("red", ax=ax_2D_plot)

    # ICE
    ice = ICE(sampler.surrogate_model, "x1", sampler.X, seed=seed)
    ice_curve_1 = ice[0]
    ice_curve_2 = ice[20]

    # ICE Curve 1
    plot_function(f, ice_curve_1.implied_config_space, ax=ax_selected_1)
    ice_curve_1.plot(line_color="red", gradient_color="lightsalmon", with_confidence=True, ax=ax_selected_1)
    plot_config_space(ice_curve_1.implied_config_space, color="red", ax=ax_2D_plot)

    # ICE Curve 2
    plot_function(f, ice_curve_2.implied_config_space, ax=ax_selected_2)
    ice_curve_2.plot(line_color="green", gradient_color="lightgreen", with_confidence=True, ax=ax_selected_2)
    plot_config_space(ice_curve_2.implied_config_space, color="green", ax=ax_2D_plot)

    # Finalize
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure 2.png")
    plt.show()


def figure_4(f: BlackboxFunction = StyblinskiTang.for_n_dimensions(2, seed=seed)):
    cs = f.config_space

    sampler = BayesianOptimizationSampler(f, cs, initial_points=f.ndim * 4, acq_class_kwargs={"tau": 0.1}, seed=seed)
    sampler.sample(50)

    surrogate = GaussianProcessSurrogate(cs, seed=seed)
    surrogate.fit(sampler.X, sampler.y)

    partitioner = DTPartitioner.from_random_points(surrogate, "x1", seed=seed)
    partitioner.partition(1)
    assert len(partitioner.leaves) == 2
    left_region, right_region = partitioner.leaves

    left_region.plot("green")
    right_region.plot("blue")
    plt.savefig("Figure 4.png")
    plt.show()


def figure_6_table_1(f_class: Type[BlackboxFunctionND] = StyblinskiTang,
                     dimensions: Iterable[int] = (3, 5, 8),
                     n_sampling_points: Iterable[int] = (80, 150, 250),
                     taus: Iterable[float] = (0.1, 1, 5),
                     n_splits: Iterable[int] = (1, 3),
                     replications: int = 30,
                     seed_offset: int = 0):
    selected_hyperparameter = "x1"
    log_file = "figure_6_table_1.csv"
    with open(log_file, "a") as d:
        d.write("{seed},{dimension},{tau},{mmd},{base_mc},{base_nll},{mc},{nll}\n".replace("{", "").replace("}", ""))
        for i in tqdm(range(replications), desc="Replication: "):
            seed = seed_offset + i
            for dimension, n_samples in zip(dimensions, n_sampling_points):
                f = f_class.for_n_dimensions(dimension, seed=seed)
                print(f)
                for tau in taus:
                    sampler = BayesianOptimizationSampler(
                        f,
                        f.config_space,
                        acq_class=LowerConfidenceBound,
                        acq_class_kwargs={"tau": tau}
                    )
                    sampler.sample(n_samples)

                    mmd = sampler.maximum_mean_discrepancy(300)

                    dt_partitioner = DTPartitioner.from_random_points(
                        surrogate_model=sampler.surrogate_model,
                        selected_hyperparameter=selected_hyperparameter,
                        seed=seed
                    )
                    base_mc = dt_partitioner.root.mean_confidence
                    base_nll = dt_partitioner.root.negative_log_likelihood(f)

                    for splits in n_splits:
                        dt_partitioner.partition(splits)
                        incumbent_region = dt_partitioner.get_incumbent_region(sampler.incumbent_config)
                        mc = incumbent_region.mean_confidence
                        nll = incumbent_region.negative_log_likelihood(f)
                        print(f"{seed=},{dimension=},{tau=},{mmd=},{base_mc=},{base_nll=},{mc=},{nll=}")
                        d.write(f"{seed},{dimension},{tau},{mmd},{base_mc},{base_nll},{mc},{nll}\n")
                        d.flush()


if __name__ == '__main__':
    figure_1_3()
    figure_2()
    figure_4()
    figure_6_table_1()

    exit(0)
    functions: List[BlackboxFunction] = [
        StyblinskiTang.for_n_dimensions(3),
        # StyblinskiTang.for_n_dimensions(5),
        # StyblinskiTang.for_n_dimensions(8),
    ]
    bo_sampling_points = [80, 150, 250]

    for f, sampling_points in zip(functions, bo_sampling_points):
        # for sampling_points in tqdm(bo_sampling_points, desc="Bayesian sampling points"):
        run_algorithm(f, f.config_space, sampling_points, n_splits=1)
