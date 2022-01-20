import unittest

import numpy as np
from ConfigSpace import Configuration
from matplotlib import pyplot as plt

from src.demo_data.blackbox_functions import square_2D
from src.demo_data.config_spaces import config_space_nd
from src.optimizer import BayesianOptimization
from src.partitioner import DecisionTreePartitioner
from src.pdp import PDP
from src.plotting import plot_ice, plot_function, plot_samples


class TestPartitioner(unittest.TestCase):
    def test_dt_partitioner_single_split(self):
        cs = config_space_nd(2)
        bo = BayesianOptimization(square_2D, config_space=cs)
        bo.optimize(10)
        selected_hp = cs.get_hyperparameters()[0]
        pdp = PDP(bo.surrogate_model,  cs)
        idx = 0
        n_samples = 1000
        n_grid_points = 20
        x_ice, y_ice, variances = pdp.calculate_ice(selected_hp, n_samples=n_samples, n_grid_points=n_grid_points)

        partitioner = DecisionTreePartitioner(idx, x_ice, variances)
        indices, means = partitioner.partition()

        self.assertTrue(np.all(np.abs(np.diff(means)) > 0))
        self.assertTrue(np.sum(indices) == n_samples)

        # plotting in different colors
        colors = ['green', 'blue']
        ax = None
        for i, index in enumerate(indices):
            ax = plot_ice(x_ice[index], y_ice[index], idx, ax=ax, color=colors[i])
        plt.show()

    def test_dt_partitioner_multiple_splits(self):
        cs = config_space_nd(2, lower=-1, upper=1)
        selected_hp = cs.get_hyperparameters()[0]
        bo = BayesianOptimization(square_2D, config_space=cs)
        bo.optimize(10)
        pdp = PDP(bo.surrogate_model,  cs)
        idx = 0
        n_samples = 1000
        n_grid_points = 20
        x_ice, y_ice, variances = pdp.calculate_ice(selected_hp, n_samples=n_samples, n_grid_points=n_grid_points)

        partitioner = DecisionTreePartitioner(idx, x_ice, variances)
        indices, means = partitioner.partition(max_depth=3)

        self.assertTrue(np.all(np.abs(np.diff(means)) > 0))
        self.assertTrue(np.sum(indices) == n_samples)

        colors = ['red', 'orange', 'green', 'blue', 'grey', 'black', 'magenta', 'yellow']
        ax = None
        for i, index in enumerate(indices):
            ax = plot_ice(x_ice[index], y_ice[index], idx, ax=ax, color=colors[i])
        plt.show()

