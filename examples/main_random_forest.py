import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from pyPDP.algorithms.ice import ICE
from pyPDP.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from pyPDP.algorithms.partitioner.random_forest_partitioner import RandomForestPartitioner
from pyPDP.blackbox_functions.synthetic_functions import StyblinskiTang
from pyPDP.sampler.bayesian_optimization import BayesianOptimizationSampler
from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from pyPDP.utils.plotting import plot_1D_confidence_lines, plot_1D_confidence_color_gradients
from pyPDP.utils.utils import calculate_log_delta

warnings.filterwarnings("ignore", category=ConvergenceWarning)

n_splits = 3
tau = 1
n_dim = 5
n_samples = 150
sample_size_tree = 200
n_initial_samples = n_dim * 4
selected_hp = 'x1'

plot_folder = Path(__file__).parent.parent / "plots" / "main_rf"
plot_folder.mkdir(parents=True, exist_ok=True)

data_folder = Path(__file__).parent.parent / 'data'
data_folder.mkdir(parents=True, exist_ok=True)


def generate_data_trees(
        num_replication: int = 10,
        log_filename: str = "rf_data.csv",
        seed_offset: int = 0
):
    num_tree_list = [1, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    # num_tree_list = [1, 2, 3]

    with (data_folder / log_filename).open("a") as file:
        file.write('seed,num_trees,base_mc,region_mc\n')

        for seed in tqdm(range(seed_offset, seed_offset + num_replication), desc='Replication: '):
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
            for num_trees in num_tree_list:
                min_incumbent_overlap = int(math.ceil(num_trees / 2))

                partitioner.partition(max_depth=n_splits, num_trees=num_trees, sample_size=sample_size_tree)
                region = partitioner.get_incumbent_region(sampler.incumbent[0],
                                                          min_incumbent_overlap=min_incumbent_overlap)

                base_mc = np.mean(np.sqrt(ice.y_variances)).item()
                region_mc = region.mean_confidence
                file.write(f'{seed},{num_trees},{base_mc},{region_mc}\n')

            # compare to decision tree
            dt_partitioner = DecisionTreePartitioner.from_ICE(ice)
            dt_partitioner.partition(n_splits)
            dt_region = dt_partitioner.get_incumbent_region(sampler.incumbent[0])
            dt_base_mc = np.mean(np.sqrt(ice.y_variances)).item()
            dt_region_mc = dt_region.mean_confidence

            file.write(f'{seed},dt,{dt_base_mc},{dt_region_mc}\n')


def plot_tree_data(log_filename: str, img_filename: str):
    df = pd.read_csv(data_folder / log_filename, header=0)

    grouped = df.groupby(df["num_trees"].astype(str))
    result_dict = {}
    for group_key, cols in grouped.groups.items():
        region_mc_vals = list(df.region_mc[cols].values)
        base_mc_vals = list(df.base_mc[cols].values)
        result_dict[group_key] = (region_mc_vals, base_mc_vals)

    plt.figure(figsize=(10, 10))

    # decision tree data point
    region_mc, base_mc = result_dict['dt']
    delta_mcs = np.asarray([
        calculate_log_delta(region_mc_val, base_mc_val)
        for region_mc_val, base_mc_val
        in zip(region_mc, base_mc)])
    dt_mean = np.mean(delta_mcs).item() * 100
    dt_std = np.std(delta_mcs).item() * 100
    x = 1
    # plt.boxplot(delta_mcs, positions=[0], manage_ticks=False)
    # plt.plot([], [], color='orange', label='Decision Tree')
    plt.plot(x, dt_mean, '*', color='red', label='Decision Tree Mean')
    plt.plot(x, dt_mean + dt_std, '*', color='orange', label=f'Decision Tree $\\mu\\pm$ $\\sigma$')
    plt.plot(x, dt_mean - dt_std, '*', color='orange')

    # rf mean curve
    x_vals = list(result_dict.keys())
    x_vals.remove('dt')
    x_vals = [int(x) for x in x_vals]
    x_vals = np.asarray(sorted(x_vals))
    y_means = []
    y_std = []
    for x in x_vals:
        region_mc, base_mc = result_dict[str(x)]
        delta_mcs = np.asarray([
            calculate_log_delta(region_mc_val, base_mc_val)
            for region_mc_val, base_mc_val
            in zip(region_mc, base_mc)])

        delta_mean = np.mean(delta_mcs).item() * 100
        delta_std = np.std(delta_mcs).item() * 100
        y_means.append(delta_mean)
        y_std.append(delta_std)
    y_means = np.asarray(y_means)
    y_std = np.asarray(y_std)

    plt.plot(x_vals, y_means, color='black', label='Random Forest Mean')
    plot_1D_confidence_lines(x_vals, y_means, stds=y_std, name='Random Forest', color='black', k_sigmas=(1,))
    plot_1D_confidence_color_gradients(x_vals, y_means, stds=y_std)
    plt.xlabel('Number of Trees')
    plt.ylabel('Delta Mean Confidence (%)')
    plt.title(f'Random Forest Splitting on: StyblinskiTang-{n_dim}D, {tau=}, {n_splits=}, {sample_size_tree=}')
    plt.legend()
    plt.savefig(plot_folder / img_filename)
    plt.show()


if __name__ == '__main__':
    log_filename = "rf_data.csv"
    img_filename = "rf_analysis.png"
    # generate_data_trees(num_replication=1, log_filename=log_filename)
    plot_tree_data(log_filename, img_filename)
