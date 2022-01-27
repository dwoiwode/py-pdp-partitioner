import numpy as np

from src.algorithms.pdp import PDP
from src.blackbox_functions import config_space_nd
from src.blackbox_functions.synthetic_functions import Square
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from test import PlottableTest


class TestPDP(PlottableTest):
    def test_calculate_pdp_1D(self):
        f = Square.for_n_dimensions(1)
        cs = f.config_space
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(f, config_space=cs)
        bo.sample(10)

        pdp = PDP(bo.surrogate_model,  selected_hp)

        self.assertTrue(pdp.x_pdp.shape[0] == pdp.y_pdp.shape[0])
        self.assertTrue(np.all(np.diff(pdp.x_pdp) > 0))

    def test_create_pdp_2D(self):
        f = Square.for_n_dimensions(2)
        cs = f.config_space
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(f, config_space=cs)
        bo.sample(10)

        pdp = PDP(bo.surrogate_model,  selected_hp)


        self.assertTrue(pdp.x_pdp.shape[0] == pdp.y_pdp.shape[0])
        self.assertTrue(pdp.x_pdp.shape[1] == 2)
        self.assertTrue(np.all(np.diff(pdp.x_pdp[:, 1]) == 0))  # value for feature not in x_s should be the same

    def test_create_pdp_centered(self):
        f = Square.for_n_dimensions(2)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs)

        selected_hp = cs.get_hyperparameter("x1")
        bo.sample(10)

        pdp = PDP(bo.surrogate_model, selected_hp)
        pdp.ice.centered = True

        self.assertTrue(pdp.y_pdp[0] == 0)




