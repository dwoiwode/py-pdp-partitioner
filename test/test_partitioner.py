import unittest

import numpy as np
from matplotlib import pyplot as plt

from src.blackbox_functions import square_2D
from src.config_spaces import square_2D_config_space
from src.optimizer import BayesianOptimization
from src.partitioner import DecisionTreePartitioner
from src.pdp import PDP
from src.plotting import plot_ice, plot_function, plot_samples


class TestPartitioner(unittest.TestCase):
    def test_dt_partitioner_single_split(self):
        bo = BayesianOptimization(square_2D, config_space=square_2D_config_space())
        bo.optimize(10)
        pdp = PDP(bo)
        idx = 0
        x_ice, y_ice, variances = pdp.calculate_ice(idx)

        partitioner = DecisionTreePartitioner(idx, x_ice, variances)
        indices, means = partitioner.partition()

        self.assertTrue(np.all(np.abs(np.diff(means)) > 0))
        self.assertTrue(np.sum(indices) == 15)

        # plotting in different colors
        colors = ['green', 'blue']
        ax = None
        for i, index in enumerate(indices):
            ax = plot_ice(x_ice[index], y_ice[index], idx, ax=ax, color=colors[i])
        plt.show()

    def test_dt_partitioner_multiple_splits(self):
        cs = square_2D_config_space()
        bo = BayesianOptimization(square_2D, config_space=cs)
        bo.optimize(10)
        pdp = PDP(bo)
        idx = 0
        x_ice, y_ice, variances = pdp.calculate_ice(idx)

        partitioner = DecisionTreePartitioner(idx, x_ice, variances)
        indices, means = partitioner.partition(max_depth=2)

        self.assertTrue(np.all(np.abs(np.diff(means)) > 0))
        self.assertTrue(np.sum(indices) == 15)

        colors = ['red', 'orange', 'green', 'blue', 'grey', 'black', 'magenta', 'yellow']
        ax = None
        for i, index in enumerate(indices):
            ax = plot_ice(x_ice[index], y_ice[index], idx, ax=ax, color=colors[i])
        plt.show()

        # create color map
        color_map = np.asarray(['This is a long string' for i in range(indices.shape[1])])
        for i, index in enumerate(indices):
            color_map[index] = colors[i]
        args = {'c': list(color_map)}
        plot_function(square_2D, cs)
        plot_samples(bo.config_list, bo.y_list, plotting_kwargs=args)
        plt.show()
