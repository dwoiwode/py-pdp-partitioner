import warnings
from pathlib import Path
from typing import Dict, Iterable, Type, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from src.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from src.algorithms.ice import ICE, ICECurve
from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.algorithms.pdp import PDP
from src.blackbox_functions import BlackboxFunction, BlackboxFunctionND
from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler import Sampler
from src.sampler.acquisition_function import LowerConfidenceBound
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.sampler.random_sampler import RandomSampler
from src.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from src.utils.plotting import plot_function, plot_config_space
from src.utils.utils import calculate_log_delta

warnings.filterwarnings("ignore", category=ConvergenceWarning)

seed = 0


def figure_1_3(f: BlackboxFunction = StyblinskiTang.for_n_dimensions(2, seed=seed),
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
        pdp.plot_values("blue", ax=ax3)
        pdp.plot_confidences("lightblue", ax=ax3)

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
    ice_curve_1.plot_values(color="red", ax=ax_selected_1)
    ice_curve_1.plot_confidences(line_color="red", gradient_color="lightsalmon", ax=ax_selected_1)
    plot_config_space(ice_curve_1.implied_config_space, color="red", ax=ax_2D_plot)

    # ICE Curve 2
    plot_function(f, ice_curve_2.implied_config_space, ax=ax_selected_2)
    ice_curve_2.plot_values(color="green", ax=ax_selected_2)
    ice_curve_2.plot_confidences(line_color="green", gradient_color="lightgreen", ax=ax_selected_2)
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

    partitioner = DecisionTreePartitioner.from_random_points(surrogate, "x1", seed=seed)
    partitioner.partition(1)
    assert len(partitioner.leaves) == 2
    left_region, right_region = partitioner.leaves

    left_region.plot("green")
    right_region.plot("blue")
    plt.savefig("Figure 4.png")
    plt.show()


def figure_6_table_1_data_generation(
        f_class: Type[BlackboxFunctionND] = StyblinskiTang,
        dimensions: Iterable[int] = (3, 5, 8),
        n_sampling_points: Iterable[int] = (80, 150, 250),
        taus: Iterable[float] = (0.1, 1, 5),
        n_splits: Iterable[int] = (1, 3),
        replications: int = 30,
        log_filename="figure6.csv",
        seed_offset: int = 0
):
    selected_hyperparameter = "x1"
    with open(log_filename, "a") as d:
        d.write("seed,dimension,tau,splits,mmd,base_mc,base_nll,mc,nll\n")
        for i in tqdm(range(replications), desc="Replication: "):
            seed = seed_offset + i
            for dimension, n_samples in zip(dimensions, n_sampling_points):
                for tau in taus:
                    f = f_class.for_n_dimensions(dimension, seed=seed)
                    print(f"{f=}, {tau=}")
                    sampler = BayesianOptimizationSampler(
                        f,
                        f.config_space,
                        initial_points=4 * dimension,
                        acq_class=LowerConfidenceBound,
                        acq_class_kwargs={"tau": tau}
                    )
                    sampler.sample(n_samples + sampler.initial_points)

                    mmd = sampler.maximum_mean_discrepancy(300)

                    surrogate = GaussianProcessSurrogate(f.config_space)
                    surrogate.fit(sampler.X, sampler.y)

                    dt_partitioner = DecisionTreePartitioner.from_random_points(
                        surrogate_model=surrogate,
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
                        print(f"{seed=},{dimension=},{tau=},{splits=},{mmd=},{base_mc=},{base_nll=},{mc=},{nll=}")
                        d.write(f"{seed},{dimension},{tau},{splits},{mmd},{base_mc},{base_nll},{mc},{nll}\n")
                        d.flush()


def figure_6_drawing(
        filename: Union[str, Path],
        columns=("base_mc", "base_nll"),
):
    df = pd.read_csv(filename, header=0)

    # Group by dimension and tau
    grouped = df.groupby(df["dimension"].astype(str) + ", " + df["tau"].astype(str))

    # Retrieve keys
    dimensions = set()
    taus = set()
    keys = {}
    for group_key in grouped.groups:
        if "dimension" in group_key:
            # There is one key named "dimension, tau". No idea how to prune it, but here we get rid of it
            continue

        dimension, tau = group_key.split(",")
        dimension = int(dimension)
        tau = float(tau)

        dimensions.add(dimension)
        taus.add(tau)
        keys[(dimension, tau)] = group_key

    n_columns = len(columns)
    column_idx = [list(df.columns).index(col) for col in columns]
    # Plot
    dimensions = sorted(dimensions)
    taus = sorted(taus)
    fig, axes = plt.subplots(
        nrows=n_columns,
        ncols=len(taus),
        sharey='row',
        sharex='all',
        figsize=(3 * len(taus), 3 * n_columns)
    )
    # gs = GridSpec(2, len(taus), figure=fig)
    for i, tau in enumerate(taus):
        axs = axes[:, i]
        # for k, col in enumerate(columns):

        plot_data = [[] for _ in range(n_columns)]
        for j, dimension in enumerate(dimensions):
            data = grouped.groups[keys[dimension, tau]]
            if len(data) == 0:
                continue

            # Get data
            for k, idx in enumerate(column_idx):
                plot_data[k].append(df.values[data, idx])

        # Plot
        ticks = list(range(1, 1 + len(dimensions)))
        for k, col in enumerate(columns):
            axs[k].boxplot(plot_data[k])
            axs[k].set_xticks(ticks, dimensions)

        # Title/X-Label
        axs[0].set_title(f"$\\tau={tau}$")
        axs[-1].set_xlabel("Dimensions")

    # Y-Label
    for k, col in enumerate(columns):
        axes[k, 0].set_ylabel(col)

    fig.savefig("Figure 6.png")
    plt.show()


def table_1_drawing(
        filename: Union[str, Path],
):
    df = pd.read_csv(filename, header=0)

    # Group by dimension and tau
    grouped = df.groupby(df["dimension"].astype(str) + ", " + df["tau"].astype(str) + ", " + df["splits"].astype(str))

    # Retrieve keys
    dimensions = set()
    taus = set()
    n_splits = set()
    keys = {}
    for group_key in grouped.groups:
        if "dimension" in group_key:
            # There is one key named "dimension, tau". No idea how to prune it, but here we get rid of it
            continue

        dimension, tau, n_split = group_key.split(",")
        dimension = int(dimension)
        tau = float(tau)
        n_split = int(n_split)

        dimensions.add(dimension)
        taus.add(tau)
        n_splits.add(n_split)
        keys[(dimension, tau, n_split)] = group_key

    list_columns = list(df.columns)
    idx_MMD = list_columns.index("mmd")
    idx_MC = list_columns.index("mc")
    idx_BASE_MC = list_columns.index("base_mc")
    idx_NLL = list_columns.index("nll")
    idx_BASE_NLL = list_columns.index("base_nll")

    # Plot
    dimensions = sorted(dimensions)
    taus = sorted(taus, reverse=True)
    n_splits = sorted(n_splits)
    # gs = GridSpec(2, len(taus), figure=fig)
    print("|                            Delta MC %         Delta NLL %")
    splits = "    |    ".join([f"{n_split}" for n_split in n_splits]) + "     |"
    print("|  d  |   Tau (MMD)    |    " + splits + "     " + splits)
    for i, dimension in enumerate(dimensions):
        for j, tau in enumerate(taus):
            table_data = [[], [], [], [], []]  # 0 = MMD, 1 = DELTA_MC, 2 = DELTA_NLL
            for n_split in n_splits:
                data = grouped.groups[keys[(dimension, tau, n_split)]]
                if len(data) == 0:
                    continue

                # Get data
                table_data[0].append(np.mean(df.values[data, idx_MMD]))
                table_data[1].append(
                    np.mean(calculate_log_delta(df.values[data, idx_MC], df.values[data, idx_BASE_MC])) * 100)
                table_data[2].append(
                    np.mean(calculate_log_delta(df.values[data, idx_NLL], df.values[data, idx_BASE_NLL])) * 100)

            string_array = [f"{mc: 6.2f}" for mc in table_data[1]]
            string_array += [f"{nll: 6.2f}" for nll in table_data[2]]
            split_data_string = "  |  ".join(string_array)
            print(f"|  {dimension}  |  {tau:.2f} ({table_data[0][0]:.2f})   | " + split_data_string)

def visualize_bad_nll():
    # worst table entry
    seed = 1
    n_splits = 3
    tau = 0.1
    n_dim = 3
    n_samples = 80
    n_initial_samples = n_dim * 4
    selected_hp = 'x1'

    f = StyblinskiTang.for_n_dimensions(n_dim, seed=seed)
    cs = f.config_space
    sampler = BayesianOptimizationSampler(f, cs, acq_class_kwargs={"tau": tau}, initial_points=n_initial_samples,
                                          seed=seed)
    sampler.sample(n_samples + n_initial_samples)

    surrogate = GaussianProcessSurrogate(cs, seed=seed)
    surrogate.fit(sampler.X, sampler.y)

    ice = ICE.from_random_points(surrogate, selected_hp)
    partitioner = DTPartitioner.from_ICE(ice)
    partitioner.partition(max_depth=n_splits)
    region = partitioner.get_incumbent_region(sampler.incumbent[0])

    ground_truth = f.pd_integral(*['x2', 'x3'], seed=seed)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        sharey='all',
        sharex='all',
        figsize=(16, 8)
    )

    # nll
    base_nll = partitioner.root.negative_log_likelihood(f)
    region_nll = region.negative_log_likelihood(f)
    delta_nll = calculate_log_delta(region_nll, base_nll)

    # original pdp
    pdp = PDP.from_ICE(ice)
    pdp.plot(ax=axes[0])
    plot_function(ground_truth, ground_truth.config_space, samples_per_axis=200, ax=axes[0])
    axes[0].legend()
    axes[0].set_title(f'Full PDP (NLL: {base_nll:.2f})')

    # split pdp
    region.plot_pdp(ax=axes[1])
    plot_function(ground_truth, ground_truth.config_space, samples_per_axis=200, ax=axes[1])
    axes[1].legend()
    axes[1].set_title(f'PDP in best Region (NLL: {region_nll:.2f})')

    plt.suptitle(f'Styblinski-Tang, 3 Splits, Tau=0.1, 3 Dimensions, (%NLL {delta_nll:.4f}), {seed=}')
    plt.show()


if __name__ == '__main__':
    figure_1_3()
    figure_2()
    figure_4()
    figure_6_table_1_data_generation(log_filename="figure_6.csv", replications=30, seed_offset=0)
    figure_6_drawing("figure_6.csv")
    table_1_drawing("figure_6.csv")
    # figure_6_table_1_drawing("figure_6_table_1.csv", columns=("base_mc", "base_nll", "mc", "nll", "mmd"))
    visualize_bad_nll()
