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
        bo = BayesianOptimization(square, config_space=square_config_space())
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        x_ice, y_ice = pdp.calculate_ice(0, ordered=False)

        self.assertTrue(len(x_ice.shape) == 3)
        self.assertTrue(len(y_ice.shape) == 2)
        self.assertTrue(x_ice.shape[0] == x_ice.shape[1])
        self.assertTrue(y_ice.shape[0] == y_ice.shape[1])
        self.assertTrue((x_ice.shape[0] == y_ice.shape[0]))
        self.assertTrue((x_ice.shape[1] == y_ice.shape[1]))

        for y_row in y_ice:
            self.assertTrue(np.all(np.abs(y_row - bo.y_list) < 0.00001))
        for y_col in y_ice.T:
            self.assertTrue(np.all(y_col == y_col[0]))

    def test_create_ice_2D(self):
        bo = BayesianOptimization(square_2D, config_space=square_2D_config_space())
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        x_ice, y_ice = pdp.calculate_ice(0, ordered=False)
        num_instances = x_ice.shape[0]

        for i in range(num_instances):
            self.assertTrue(np.all(x_ice[:, i, 0] == x_ice[0, i, 0]))
            self.assertTrue(np.all(x_ice[i, :, 1] == x_ice[i, 0, 1]))

        for i in range(num_instances):
            self.assertTrue(y_ice[i, i] - bo.y_list[i] < 0.00001)

    def test_create_ice_ordered(self):
        bo = BayesianOptimization(square_2D, config_space=square_2D_config_space())
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        x_ice, y_ice = pdp.calculate_ice(0, ordered=True)
        num_instances = x_ice.shape[0]

        # assert x_ice is sorted ascending for x_s
        for i in range(num_instances):
            self.assertTrue(np.all(np.diff(x_ice[i, :, 0]) > 0))

    def test_create_ice_centered(self):
        bo = BayesianOptimization(square_2D, config_space=square_2D_config_space())
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        x_ice, y_ice = pdp.calculate_ice(0, ordered=True, centered=True)

        self.assertTrue(np.all(y_ice[:, 0] == 0))











