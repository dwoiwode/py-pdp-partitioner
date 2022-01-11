import unittest

import numpy as np

from src.blackbox_functions import square, square_2D
from src.config_spaces import square_config_space
from src.optimizer import BayesianOptimization
from src.partitioner import DecisionTreePartitioner
from src.pdp import PDP


class TestICE(unittest.TestCase):
    def test_create_ice_1D(self):
        bo = BayesianOptimization(square, config_space=square_config_space())
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        X_ice, y_ice = pdp.calculate_ice(0)
        for y_row in y_ice.T:
            self.assertTrue(np.all(np.abs(y_row - bo.y_list) < 0.0001))

    def test_create_ice_2D(self):
        bo = BayesianOptimization(square_2D, config_space=square_2D_config_space())
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        X_ice, y_ice = pdp.calculate_ice(0)
        for y_row in y_ice.T:
            self.assertTrue(np.all(np.abs(y_row - bo.y_list) < 0.0001))


