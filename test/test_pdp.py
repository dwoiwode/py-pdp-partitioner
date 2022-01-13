import unittest

import numpy as np

from src.blackbox_functions import square, square_2D
from src.config_spaces import square_config_space, square_2D_config_space
from src.optimizer import BayesianOptimization
from src.partitioner import DecisionTreePartitioner
from src.pdp import PDP
from src.utils import config_list_to_2d_arr


class TestPDP(unittest.TestCase):
    def test_calculate_pdp_1D(self):
        bo = BayesianOptimization(square, config_space=square_config_space())
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        x_pdp, y_pdp, stds = pdp.calculate_pdp(0)

        self.assertTrue(x_pdp.shape[0] == y_pdp.shape[0])
        self.assertTrue(np.all(np.diff(x_pdp) > 0))

        x_original = np.sort(config_list_to_2d_arr(bo.config_list), axis=0)
        self.assertTrue(np.all(np.abs(x_pdp - x_original) < 0.00001))

    def test_create_pdp_2D(self):
        bo = BayesianOptimization(square_2D, config_space=square_2D_config_space())
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        x_pdp, y_pdp, stds = pdp.calculate_pdp(0, centered=False)

        self.assertTrue(x_pdp.shape[0] == y_pdp.shape[0])
        self.assertTrue(x_pdp.shape[1] == 2)
        self.assertTrue(np.all(np.diff(x_pdp[:, 1]) == 0))  # value for feature not in x_s should be the same

    def test_create_pdp_centered(self):
        bo = BayesianOptimization(square_2D, config_space=square_2D_config_space())
        partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(partitioner, bo)
        x_pdp, y_pdp, stds = pdp.calculate_pdp(0, centered=True)

        self.assertTrue(y_pdp[0] == 0)




