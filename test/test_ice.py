import unittest

import numpy as np
from matplotlib import pyplot as plt

from src.blackbox_functions import square, square_2D
from src.config_spaces import square_config_space, square_2D_config_space
from src.optimizer import BayesianOptimization
from src.partitioner import DecisionTreePartitioner
from src.pdp import PDP
from src.plotting import plot_ice
from src.utils import unscale


class TestICE(unittest.TestCase):
    def test_create_ice_1D(self):
        cs = square_config_space()
        bo = BayesianOptimization(square, config_space=cs)
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        num_grid_points = 1000
        x_ice, y_ice, variances = pdp.calculate_ice(0, num_grid_points=num_grid_points)
        num_instances = x_ice.shape[0]

        self.assertTrue(len(x_ice.shape) == 3)
        self.assertTrue(len(y_ice.shape) == 2)
        self.assertTrue(y_ice.shape == variances.shape)
        self.assertTrue(x_ice.shape[1] == num_grid_points)
        self.assertTrue(y_ice.shape[1] == num_grid_points)
        self.assertTrue((x_ice.shape[0] == y_ice.shape[0]))
        self.assertTrue((x_ice.shape[1] == y_ice.shape[1]))

        self.assertTrue(np.max(x_ice) == 1)
        self.assertTrue(np.min(x_ice) == 0)
        self.assertTrue(np.all(x_ice[:, 0, 0] == 0))
        self.assertTrue(np.all(x_ice[:, -1, 0] == 1))

        for i in range(num_instances):  # x should be ordered in x_0
            self.assertTrue(np.all(np.diff(x_ice[i, :, 0]) > 0))

    def test_create_ice_2D(self):
        bo = BayesianOptimization(square_2D, config_space=square_2D_config_space())
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        num_grid_points = 1000
        x_ice, y_ice, stds = pdp.calculate_ice(0, num_grid_points=num_grid_points)
        num_instances = x_ice.shape[0]

        self.assertTrue(x_ice.shape[0] == num_instances)
        self.assertTrue(x_ice.shape[1] == num_grid_points)
        self.assertTrue(x_ice.shape[2] == 2)

        for i in range(num_grid_points):  # x_0 should be the same across all instances
            self.assertTrue(np.all(x_ice[:, i, 0] == x_ice[0, i, 0]))

        for i in range(num_instances):  # x_1 should be the same for all x_0
            self.assertTrue(np.all(np.diff(x_ice[i, :, 1]) == 0))

    def test_create_ice_centered(self):
        bo = BayesianOptimization(square_2D, config_space=square_2D_config_space())
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        x_ice, y_ice, stds = pdp.calculate_ice(0, centered=True)

        self.assertTrue(np.all(y_ice[:, 0] == 0))











