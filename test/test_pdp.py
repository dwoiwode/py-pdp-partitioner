import unittest

import numpy as np

from src.demo_data.blackbox_functions import square, square_2D
from src.demo_data.config_spaces import config_space_nd
from src.sampler import BayesianOptimizationSampler
from src.algorithms.pdp import PDP


class TestPDP(unittest.TestCase):
    def test_calculate_pdp_1D(self):
        cs = config_space_nd(1)
        bo = BayesianOptimizationSampler(square, config_space=cs)
        selected_hp = cs.get_hyperparameters()[0]
        bo.sample(10)
        pdp = PDP(bo.surrogate_model,  cs)
        x_pdp, y_pdp, stds = pdp.calculate_pdp(selected_hp)

        self.assertTrue(x_pdp.shape[0] == y_pdp.shape[0])
        self.assertTrue(np.all(np.diff(x_pdp) > 0))

    def test_create_pdp_2D(self):
        cs = config_space_nd(2)
        bo = BayesianOptimizationSampler(square_2D, config_space=cs)
        selected_hp = cs.get_hyperparameters()[0]
        bo.sample(10)
        pdp = PDP(bo.surrogate_model,  cs)
        x_pdp, y_pdp, stds = pdp.calculate_pdp(selected_hp, centered=False)

        self.assertTrue(x_pdp.shape[0] == y_pdp.shape[0])
        self.assertTrue(x_pdp.shape[1] == 2)
        self.assertTrue(np.all(np.diff(x_pdp[:, 1]) == 0))  # value for feature not in x_s should be the same

    def test_create_pdp_centered(self):
        cs = config_space_nd(2)
        bo = BayesianOptimizationSampler(square_2D, config_space=cs)
        selected_hp = cs.get_hyperparameters()[0]
        bo.sample(10)
        pdp = PDP(bo.surrogate_model,  cs)
        x_pdp, y_pdp, stds = pdp.calculate_pdp(selected_hp, centered=True)

        self.assertTrue(y_pdp[0] == 0)




