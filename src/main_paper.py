import warnings
from pathlib import Path
from typing import Dict, Iterable, Type, Union

import pandas as pd
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
        d.write(
            "{seed},{dimension},{tau},{splits},{mmd},{base_mc},{base_nll},{mc},{nll}\n".replace("{", "").replace("}",
                                                                                                                 ""))
        for i in tqdm(range(replications), desc="Replication: "):
            seed = seed_offset + i
            for dimension, n_samples in zip(dimensions, n_sampling_points):
                for tau in taus:
                    f = f_class.for_n_dimensions(dimension, seed=seed)
                    print(f"{f=}, {tau=}")
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
                        print(f"{seed=},{dimension=},{tau=},{splits=}{mmd=},{base_mc=},{base_nll=},{mc=},{nll=}")
                        d.write(f"{seed},{dimension},{tau},{splits},{mmd},{base_mc},{base_nll},{mc},{nll}\n")
                        d.flush()


def figure_6_table_1_drawing(
        filename:Union[str, Path],
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
        axs = axes[:,i]
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


if __name__ == '__main__':
    figure_1_3()
    figure_2()
    figure_4()
    # figure_6_table_1_data_generation(filename="figure_6_table_1.csv")
    figure_6_table_1_drawing("figure_6_table_1.csv")
    # figure_6_table_1_drawing("figure_6_table_1.csv", columns=("base_mc", "base_nll", "mc", "nll", "mmd"))
