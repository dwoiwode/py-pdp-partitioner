import warnings
from pathlib import Path
from typing import Type

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import Matern

from pyPDP.algorithms.partitioner import Partitioner
from pyPDP.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from pyPDP.blackbox_functions.synthetic_functions import StyblinskiTang
from pyPDP.sampler.acquisition_function import LowerConfidenceBound
from pyPDP.sampler.bayesian_optimization import BayesianOptimizationSampler
from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from pyPDP.utils.advanced_plots import plot_hyperparameter_array_2D
from pyPDP.utils.utils import calculate_log_delta

warnings.filterwarnings("ignore", category=ConvergenceWarning)

plot_folder = Path(__file__).parent.parent / "plots" / "main_recursion"
plot_folder.mkdir(parents=True, exist_ok=True)

FIG_RES = 4
N_REPETITIONS = 1
seed = 0


def blackbox_function(
        tau: float = 1,
        n_splits: int = 3,
        n_dim: int = 3,
        kernel_size: float = 1.5,
        partitioner_class: Type[Partitioner] = DecisionTreePartitioner
) -> float:
    n_dim = int(n_dim)
    f = StyblinskiTang.for_n_dimensions(n_dim, seed=seed)
    n_samples = 34 * n_dim - 20  # Approx paper values
    log_deltas = []
    for s in range(seed, seed + N_REPETITIONS):
        surrogate = GaussianProcessSurrogate(f.config_space, seed=seed, kernel=Matern(nu=kernel_size))
        sampler = BayesianOptimizationSampler(
            f,
            f.config_space,
            surrogate_model=surrogate,
            initial_points=4 * n_dim,
            acq_class=LowerConfidenceBound,
            acq_class_kwargs={"tau": tau},
            seed=seed
        )
        sampler.sample(n_samples + sampler.initial_points)

        surrogate.fit(sampler.X, sampler.y)

        dt_partitioner = partitioner_class.from_random_points(
            surrogate_model=surrogate,
            selected_hyperparameter="x1",
            seed=seed
        )
        base_mc = dt_partitioner.root.mean_confidence

        dt_partitioner.partition(n_splits)
        incumbent_region = dt_partitioner.get_incumbent_region(sampler.incumbent_config)
        mc = incumbent_region.mean_confidence

        log_deltas.append(calculate_log_delta(mc, base_mc))

    return - np.mean(log_deltas)  # We want to maximize value -> minimize negative value


def optimize_mc():
    # Initialize
    hyperparameters = [
        # CSH.UniformIntegerHyperparameter("n_dim", 2, 15),
        # CSH.UniformFloatHyperparameter("tau", 1e-5, 10, log=True),
        # CSH.UniformIntegerHyperparameter("n_splits", 1, 10),
        CSH.UniformFloatHyperparameter("kernel_size", 0.1, 5)
        # CSH.CategoricalHyperparameter("partitioner", choices=[DecisionTreePartitioner, RandomForestPartitioner])
    ]
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add_hyperparameters(hyperparameters)

    f = blackbox_function

    # Optimize
    n_dim = len(cs.get_hyperparameters())
    sampler = BayesianOptimizationSampler(
        f,
        cs,
        initial_points=4 * n_dim,
        acq_class=LowerConfidenceBound,
        acq_class_kwargs={"tau": 0.1},
        seed=seed
    )
    sampler.sample(34 * n_dim - 20, show_progress=True)
    sampler.save_cache()
    sampler.plot()
    plt.tight_layout()
    plt.savefig(plot_folder / "samples")
    plt.show()

    surrogate = GaussianProcessSurrogate(cs, seed=seed)
    surrogate.fit(sampler.X, sampler.y)

    fig_mean, fig_std = plot_hyperparameter_array_2D(surrogate, num_samples=2000, num_grid_points_per_axis=50)

    fig_mean.savefig(plot_folder / "hyperparameters_mean")
    fig_std.savefig(plot_folder / "hyperparameters_std")
    plt.show()


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    optimize_mc()
