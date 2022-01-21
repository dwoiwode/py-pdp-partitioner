from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.demo_data.blackbox_functions import square_2D
from src.demo_data.config_spaces import config_space_nd
from src.algorithms.pdp import PDP
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from test import PlottableTest


class TestPartitioner(PlottableTest):
    def test_dt_partitioner_single_split(self):
        self.initialize_figure()
        cs = config_space_nd(2)
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(square_2D, config_space=cs)
        bo.sample(10)

        ice = ICE(bo.surrogate_model,  selected_hp)
        partitioner = DTPartitioner.from_ICE(ice)
        indices, means = partitioner.partition()
        from sklearn.inspection import plot_partial_dependence
        plot_partial_dependence()

        self.assertTrue(np.all(np.abs(np.diff(means)) > 0))
        self.assertTrue(np.sum(indices) == ice.num_samples)

        # plotting in different colors
        colors = ['green', 'blue']
        ax = None
        for i, index in enumerate(indices):
            ax = plot_ice(x_ice[index], y_ice[index], idx, ax=ax, color=colors[i])
        plt.show()

    def test_dt_partitioner_multiple_splits(self):
        cs = config_space_nd(2, lower=-1, upper=1)
        selected_hp = cs.get_hyperparameters()[0]

        bo = BayesianOptimizationSampler(square_2D, config_space=cs)
        bo.sample(10)
        ice = ICE(bo.surrogate_model,  selected_hp)

        partitioner = DTPartitioner.from_ICE(ice)
        indices, means = partitioner.partition(max_depth=3)

        self.assertTrue(np.all(np.abs(np.diff(means)) > 0))
        self.assertTrue(np.sum(indices) == ice.num_samples)

        colors = ['red', 'orange', 'green', 'blue', 'grey', 'black', 'magenta', 'yellow']
        ax = None
        for i, index in enumerate(indices):
            ax = plot_ice(x_ice[index], y_ice[index], idx, ax=ax, color=colors[i])
        plt.show()

