from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt

from src.blackbox_functions import BlackboxFunctionND, BlackboxFunction
from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler.acquisition_function import LowerConfidenceBound
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from src.utils.plotting import plot_function

RESULT_FOLDER = Path("../plots")
SEED = 29
DRAFT = True

if DRAFT:
    PLOT_SAMPLES_PER_AXIS = 50
    NUM_GRID_POINTS = 20
else:
    PLOT_SAMPLES_PER_AXIS = 200
    NUM_GRID_POINTS = 50  # 100 might be even better. But pretty memory hungry


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
    legend_location = "upper left"
    fig = plt.figure(figsize=(16, 9))
    ax = fig.gca()
    sampler = BayesianOptimizationSampler(
        obj_func=f,
        config_space=f.config_space,
        initial_points=4 * f.ndim,
        acq_class=acquisition_function,
        acq_class_kwargs=acq_kwargs,
        seed=SEED
    )

    # Blackbox function
    sampler.sample(sampler.initial_points)
    plot_function(f, f.config_space, samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax)
    ax.legend(loc=legend_location)
    yield fig, "blackbox_function"
    init_ymin, init_ymax = ax.get_ylim()

    # Initial Samples
    fig = plt.figure(figsize=(16, 9))
    ax = fig.gca()
    plot_function(f, f.config_space, samples_per_axis=PLOT_SAMPLES_PER_AXIS,color=(0.7, 0.7, 0.7), ax=ax)
    sampler.plot(ax=ax)
    ax.legend(loc=legend_location)
    yield fig, "samples"

    # Surrogate
    surrogate = GaussianProcessSurrogate(f.config_space, seed=SEED)
    surrogate.fit(sampler.X, sampler.y)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.gca()
    # Plot f
    plot_function(f, f.config_space, color=(0.7, 0.7, 0.7), ax=ax)
    # Plot Surrogate
    surrogate.plot_means(samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax)
    surrogate.plot_confidences(samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax)
    # Plot Sampler
    sampler.plot(ax=ax)

    # Finalize plot
    ax.legend(loc=legend_location)
    yield fig, "surrogate"

    for i in range(5):
        sampler.sample(1)
        surrogate.fit(sampler.X, sampler.y)

        fig = plt.figure(figsize=(16, 9))
        ax = fig.gca()
        # Plot f
        plot_function(f, f.config_space, color=(0.7, 0.7, 0.7), ax=ax)
        # Plot Surrogate
        surrogate.plot_means(samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax)
        surrogate.plot_confidences(samples_per_axis=PLOT_SAMPLES_PER_AXIS, ax=ax)
        # Plot Sampler
        sampler.plot(ax=ax)

        ax.legend(loc=legend_location)
        ax.set_ylim(init_ymin, init_ymax)
        yield fig, "surrogate"


def algorithm_walkthrough(
        f: BlackboxFunction = StyblinskiTang.for_n_dimensions(2, seed=SEED),
        acquisition_function=LowerConfidenceBound,
        acq_kwargs=None,

):
    folder = RESULT_FOLDER / "algorithm_walkthrough"
    folder.mkdir(exist_ok=True, parents=True)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.gca()
    sampler = BayesianOptimizationSampler(
        obj_func=f,
        config_space=f.config_space,
        initial_points=4 * f.ndim,
        acq_class=acquisition_function,
        acq_class_kwargs=acq_kwargs,
        seed=SEED
    )

    sampler.sample(sampler.initial_points)
    plot_function(f, f.config_space, ax=ax)
    fig.savefig(folder / "00_blackbox_function.png")


if __name__ == '__main__':
    bo_introduction()
    # algorithm_walkthrough()
