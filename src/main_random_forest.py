import warnings
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from src.algorithms.partitioner.random_forest_partitioner import RandomForestPartitioner
from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from src.utils.utils import calculate_log_delta

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def analyze_num_trees(  # Hyperparameter Analysis: Number of Trees
        num_replication: int = 10,
        num_tree_list: List[int] = None):
    if num_tree_list is None:
        num_tree_list = [1, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
        # num_tree_list = [1, 2, 3]
    n_splits = 3
    tau = 1
    n_dim = 3
    n_samples = 80
    sample_size_tree = 200
    n_initial_samples = n_dim * 4
    selected_hp = 'x1'

    delta_mc_arr = np.zeros_like(num_tree_list, dtype=float)
    dt_delta_mc = 0
    for i in range(num_replication):
        seed = i
        f = StyblinskiTang.for_n_dimensions(n_dim, seed=seed)
        cs = f.config_space
        sampler = BayesianOptimizationSampler(f, cs, acq_class_kwargs={"tau": tau},
                                              initial_points=n_initial_samples,
                                              seed=seed)
        sampler.sample(n_samples + n_initial_samples)

        surrogate = GaussianProcessSurrogate(cs, seed=seed)
        surrogate.fit(sampler.X, sampler.y)

        ice = ICE.from_random_points(surrogate, selected_hp)
        partitioner = RandomForestPartitioner.from_ICE(ice, seed=seed)

        # random forest
        for idx, num_trees in enumerate(num_tree_list):
            min_incumbent_overlap = int(num_trees / 2)

            partitioner.partition(max_depth=n_splits, num_trees=num_trees, sample_size=sample_size_tree)
            region = partitioner.get_incumbent_region(sampler.incumbent[0], min_incumbent_overlap=min_incumbent_overlap)

            base_mc = np.mean(np.sqrt(ice.y_variances)).item()
            region_mc = region.mean_confidence
            delta_mc = calculate_log_delta(region_mc, base_mc)
            delta_mc_arr[idx] = delta_mc_arr[idx] + delta_mc

        # compare to decision tree
        dt_partitioner = DecisionTreePartitioner.from_ICE(ice)
        dt_partitioner.partition(n_splits)
        dt_region = dt_partitioner.get_incumbent_region(sampler.incumbent[0])
        base_mc = np.mean(np.sqrt(ice.y_variances)).item()
        dt_region_mc = dt_region.mean_confidence
        dt_delta_mc += calculate_log_delta(dt_region_mc, base_mc)

    dt_delta_mc *= (100 / num_replication)
    delta_mc_arr = delta_mc_arr * 100
    delta_mc_arr = delta_mc_arr / num_replication

    # plotting
    plt.figure(figsize=(10, 10))
    plt.plot(num_tree_list, delta_mc_arr, label='Random Forest')
    plt.xlabel('Number of Trees')
    plt.ylabel('Delta Mean Confidence (%)')
    plt.title(f'Random Forest Splitting on: Styblinski-Tang-3D, {tau=}, {n_splits=}, {sample_size_tree=}')
    plt.plot(1, dt_delta_mc, "*", color='red', label='Decision Tree', markersize=15)
    plt.legend()
    plt.savefig('../results/random_forest/RF_Hyperparameter_analysis_num_trees')


if __name__ == '__main__':
    analyze_num_trees()
