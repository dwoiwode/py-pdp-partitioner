import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.colors
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning

from pyPDP.algorithms.ice import ICE
from pyPDP.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from pyPDP.algorithms.pdp import PDP
from pyPDP.blackbox_functions import BlackboxFunctionND, BlackboxFunction
from pyPDP.blackbox_functions.synthetic_functions import StyblinskiTang
from pyPDP.sampler.acquisition_function import LowerConfidenceBound
from pyPDP.sampler.bayesian_optimization import BayesianOptimizationSampler
from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from pyPDP.utils.plotting import plot_function, plot_config_space

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULT_FOLDER = Path("../plots")
RANDOM_COLORS = tuple(matplotlib.colors.BASE_COLORS.values())
SEED = 31  # Good seeds: 29
DRAFT = True

if DRAFT:
    PLOT_SAMPLES_PER_AXIS = 50
    NUM_GRID_POINTS = 20
else:
    PLOT_SAMPLES_PER_AXIS = 300
    NUM_GRID_POINTS = 70  # 100 might be even better. But pretty memory hungry


def save_yielded_plots(folder: Optional[str] = None):
    def save_plots(f):
        if folder is None:
            result_folder = RESULT_FOLDER / f.__name__
        else:
            result_folder = RESULT_FOLDER / folder
        result_folder.mkdir(exist_ok=True, parents=True)

        def wrapped(*args, **kwargs):
            for i, (fig, fig_name) in enumerate(f(*args, **kwargs)):
                filename = result_folder / f"{i:02d}_{fig_name}.png"
                fig.tight_layout()
                print(f"Saving figure {filename}")
                fig.savefig(filename)
            plt.close("all")

        return wrapped

    return save_plots


@save_yielded_plots()
def bo_introduction(
        f: BlackboxFunction = StyblinskiTang.for_n_dimensions(1, seed=SEED),
        acquisition_function=LowerConfidenceBound,
        acq_kwargs=None,

):
    def new_figure() -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        fig = plt.figure(figsize=(16, 9))
        assert isinstance(fig, plt.Figure)

        ax_function = fig.add_subplot(3, 1, (1, 2))
        ax_acquisition = fig.add_subplot(3, 1, 3)
        assert isinstance(ax_function, plt.Axes)
        assert isinstance(ax_acquisition, plt.Axes)
        return fig, (ax_function, ax_acquisition)

    legend_location = "upper left"
    sampler = BayesianOptimizationSampler(
        obj_func=f,
        config_space=f.config_space,
        initial_points=4 * f.ndim,
        acq_class=acquisition_function,
        acq_class_kwargs=acq_kwargs,
        seed=SEED
    )
    sampler.sample(sampler.initial_points)

    # Blackbox function
    fig, (ax, _) = new_figure()
    plot_function(f, f.config_space, samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax)
    ax.legend(loc=legend_location)
    yield fig, "blackbox_function"
    init_ymin, init_ymax = ax.get_ylim()
    init_ymin *= 0.9
    init_ymax *= 1.1

    # Initial Samples
    fig, (ax, _) = new_figure()
    plot_function(f, f.config_space, samples_per_axis=PLOT_SAMPLES_PER_AXIS, color=(0.7, 0.7, 0.7), ax=ax)
    sampler.plot(marker="*", ax=ax)
    ax.legend(loc=legend_location)
    yield fig, "samples"

    # Surrogate
    surrogate = GaussianProcessSurrogate(f.config_space, seed=SEED)
    surrogate.fit(sampler.X, sampler.y)

    fig, (ax_f, ax_acq) = new_figure()
    # Plot f
    plot_function(f, f.config_space, color=(0.7, 0.7, 0.7), ax=ax_f)
    # Plot Surrogate
    surrogate.plot_means(samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax_f)
    surrogate.plot_confidences(samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax_f)
    # Plot Sampler
    sampler.plot(ax=ax_f)

    # Finalize plot
    ax_f.legend(loc=legend_location)
    yield fig, "surrogate"
    sampler.acq_func.plot(ax=ax_acq)
    yield fig, "acquisition"

    for i in range(5):
        sampler.sample(1)
        surrogate.fit(sampler.X, sampler.y)

        fig, (ax_f, ax_acq) = new_figure()
        # Plot f
        plot_function(f, f.config_space, color=(0.7, 0.7, 0.7), ax=ax_f)
        # Plot Surrogate
        surrogate.plot_means(samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax_f)
        surrogate.plot_confidences(samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax_f)
        # Plot Sampler
        sampler.plot(ax=ax_f)

        ax_f.legend(loc=legend_location)
        ax_f.set_ylim(init_ymin, init_ymax)
        sampler.acq_func.plot(ax=ax_acq)
        yield fig, "acquisition"

    # Sample a lot
    sampler.sample(51)  # Total of 60 Points
    surrogate.fit(sampler.X, sampler.y)

    fig, (ax_f, ax_acq) = new_figure()
    # Plot f
    plot_function(f, f.config_space, color=(0.7, 0.7, 0.7), ax=ax_f)
    # Plot Surrogate
    surrogate.plot_means(samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax_f)
    surrogate.plot_confidences(samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax_f)
    # Plot Sampler
    sampler.plot(ax=ax_f)

    ax_f.legend(loc=legend_location)
    ax_f.set_ylim(init_ymin, init_ymax)
    sampler.acq_func.plot(ax=ax_acq)
    yield fig, "acquisition"


@save_yielded_plots()
def algorithm_walkthrough(
        f: BlackboxFunction = StyblinskiTang.for_n_dimensions(2, seed=SEED),
        acquisition_function=LowerConfidenceBound,
        acq_kwargs=None,

):
    def new_figure() -> Tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(8, 6), dpi=200)
        ax = fig.gca()
        return fig, ax

    fig, ax = new_figure()
    # Sampler
    sampler = BayesianOptimizationSampler(
        obj_func=f,
        config_space=f.config_space,
        initial_points=4 * f.ndim,
        acq_class=acquisition_function,
        acq_class_kwargs=acq_kwargs,
        seed=SEED
    )

    sampler.sample(sampler.initial_points)
    plot_function(f, f.config_space, ax=ax, samples_per_axis=PLOT_SAMPLES_PER_AXIS)
    yield fig, "Blackboxfunction"
    sampler.plot(ax=ax)
    yield fig, "Sampler"

    sampler.sample(50)
    sampler.plot(ax=ax)
    yield fig, "Sampler"

    # Surrogate
    surrogate_model = GaussianProcessSurrogate(cs=f.config_space, seed=SEED)
    surrogate_model.fit(sampler.X, sampler.y)
    fig, ax = new_figure()
    surrogate_model.plot_means(ax=ax, samples_per_axis=PLOT_SAMPLES_PER_AXIS)
    yield fig, "Surrogate_Means"

    fig, ax = new_figure()
    surrogate_model.plot_confidences(ax=ax, samples_per_axis=PLOT_SAMPLES_PER_AXIS)
    yield fig, "Surrogate_Confidences"

    # ICE
    ice = ICE.from_random_points(surrogate_model, "x1", seed=SEED, num_grid_points_per_axis=NUM_GRID_POINTS)
    fig, ax = new_figure()
    ice.plot(ax=ax)
    yield fig, "ICE"

    # PDP
    pdp = PDP.from_ICE(ice, seed=SEED)
    # fig, ax = new_figure()
    pdp.plot_values(ax=ax, color="black")
    yield fig, "PDP"
    pdp.plot_confidences(ax=ax, line_color="black", gradient_color="black")
    pdp.plot_values(ax=ax, color="black")
    yield fig, "PDP_with_confidence"

    # Decision Tree Partitioner
    fig, ax = new_figure()
    dt_partitioner = DecisionTreePartitioner.from_ICE(ice)
    dt_partitioner.partition(3)
    dt_partitioner.plot(ax=ax)
    yield fig, "DecisionTreePartitioner"

    # Plot all regions
    fig, ax = new_figure()
    plot_function(f, f.config_space, samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax)
    for i, leaf in enumerate(dt_partitioner.leaves):
        plot_config_space(leaf.implied_config_space(seed=SEED), color=RANDOM_COLORS[i], alpha=0.3)
        ax.annotate(f"{leaf.mean_confidence}")

    sampler.plot(ax=ax)
    yield fig, "DT All Regions"

    # Plot incumbent region
    fig, ax = new_figure()
    plot_function(f, f.config_space, samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax)
    incumbent_region = dt_partitioner.get_incumbent_region(sampler.incumbent_config)
    plot_config_space(incumbent_region.implied_config_space(seed=SEED))
    yield fig, "DT Incumbent Region"


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.NOTSET)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    bo_introduction()
    algorithm_walkthrough()
