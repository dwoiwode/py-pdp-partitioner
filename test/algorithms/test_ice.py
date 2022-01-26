import unittest

import numpy as np

from src.algorithms.ice import ICE
from src.blackbox_functions.synthetic_functions import Square
from src.sampler.bayesian_optimization import BayesianOptimizationSampler


class TestICE(unittest.TestCase):
    def test_create_ice_1D(self):
        f = Square(1)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs)
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
        f = Square(2)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs)
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
        f = Square(2)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs)
        selected_hp = cs.get_hyperparameters()[0]

        bo.sample(10)
        ice = ICE(bo.surrogate_model, selected_hp)
        ice.centered = True
        y_ice = ice.y_ice

        self.assertTrue(np.all(y_ice[:, 0] == 0))
