import unittest

import numpy as np

from src.algorithms.ice import ICE
from src.demo_data.blackbox_functions import square, square_2D
from src.demo_data.config_spaces import config_space_nd
from src.sampler.bayesian_optimization import BayesianOptimizationSampler


class TestICE(unittest.TestCase):
    def test_create_ice_1D(self):
        cs = config_space_nd(1)
        bo = BayesianOptimizationSampler(square, config_space=cs)
        selected_hp = cs.get_hyperparameter("x1")

        bo.sample(10)
        num_grid_points = 1000
        ice = ICE(bo.surrogate_model, selected_hp, num_grid_points_per_axis=num_grid_points)
        x_ice = ice.x_ice
        y_ice = ice.y_ice
        variances = ice.y_variances

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
        cs = config_space_nd(2)
        bo = BayesianOptimizationSampler(square_2D, config_space=cs)
        selected_hyperparameter = cs.get_hyperparameter("x1")

        bo.sample(10)
        num_grid_points = 20
        n_samples = 1000
        ice = ICE(bo.surrogate_model, selected_hyperparameter,
                  num_samples=n_samples, num_grid_points_per_axis=num_grid_points)

        x_ice = ice.x_ice
        y_ice = ice.y_ice
        variances = ice.y_variances

        self.assertTrue(x_ice.shape[0] == n_samples)
        self.assertTrue(x_ice.shape[1] == num_grid_points)
        self.assertTrue(x_ice.shape[2] == 2)

        for i in range(num_grid_points):  # x_0 should be the same across all instances
            self.assertTrue(np.all(x_ice[:, i, 0] == x_ice[0, i, 0]))

        for i in range(n_samples):  # x_1 should be the same for all x_0
            self.assertTrue(np.all(np.diff(x_ice[i, :, 1]) == 0))

    def test_create_ice_centered(self):
        cs = config_space_nd(2)
        bo = BayesianOptimizationSampler(square_2D, config_space=cs)
        selected_hp = cs.get_hyperparameters()[0]

        bo.sample(10)
        ice = ICE(bo.surrogate_model, selected_hp)
        ice.centered = True
        y_ice = ice.y_ice

        self.assertTrue(np.all(y_ice[:, 0] == 0))
