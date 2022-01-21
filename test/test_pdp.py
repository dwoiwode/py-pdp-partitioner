import numpy as np

from src.algorithms.pdp import PDP
from src.demo_data.blackbox_functions import square, square_2D
from src.demo_data.config_spaces import config_space_nd
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from test import PlottableTest


class TestPDP(PlottableTest):
    def test_calculate_pdp_1D(self):
        cs = config_space_nd(1)
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(square, config_space=cs)
        bo.sample(10)

        pdp = PDP(bo.surrogate_model,  selected_hp)

        self.assertTrue(pdp.x_pdp.shape[0] == pdp.y_pdp.shape[0])
        self.assertTrue(np.all(np.diff(pdp.x_pdp) > 0))

    def test_create_pdp_2D(self):
        cs = config_space_nd(2)
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(square_2D, cs)
        bo.sample(10)

        pdp = PDP(bo.surrogate_model,  selected_hp)


        self.assertTrue(pdp.x_pdp.shape[0] == pdp.y_pdp.shape[0])
        self.assertTrue(pdp.x_pdp.shape[1] == 2)
        self.assertTrue(np.all(np.diff(pdp.x_pdp[:, 1]) == 0))  # value for feature not in x_s should be the same

    def test_create_pdp_centered(self):
        cs = config_space_nd(2)
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(square_2D, config_space=cs)
        bo.sample(10)

        pdp = PDP(bo.surrogate_model, selected_hp)
        pdp.ice.centered = True

        self.assertTrue(pdp.y_pdp[0] == 0)




