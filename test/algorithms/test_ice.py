import ConfigSpace.hyperparameters as CSH
import numpy as np

from src.algorithms.ice import ICE
from src.blackbox_functions.synthetic_functions import Square, StyblinskiTang
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from test import PlottableTest


class TestICE(PlottableTest):
    def test_create_ice_1D_f_1D(self):
        f = Square.for_n_dimensions(1)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs)
        selected_hp = cs.get_hyperparameter("x1")

        bo.sample(10)
        num_grid_points = 1000
        ice = ICE.from_random_points(bo.surrogate_model, selected_hp, num_grid_points_per_axis=num_grid_points)
        x_ice = ice.x_ice
        y_ice = ice.y_ice
        variances = ice.y_variances
        grid_points = ice.grid_points

        num_instances = x_ice.shape[0]

        self.assertEqual(3, len(x_ice.shape))
        self.assertEqual(2, len(y_ice.shape))
        self.assertTrue(y_ice.shape == variances.shape)
        self.assertEqual(x_ice.shape[1], num_grid_points)
        self.assertEqual(y_ice.shape[1], num_grid_points)
        self.assertEqual(x_ice.shape[0], y_ice.shape[0])
        self.assertEqual(x_ice.shape[1], y_ice.shape[1])
        self.assertEqual(num_grid_points, grid_points.shape[0])
        self.assertEqual(1, grid_points.shape[1])

        self.assertEqual(1, np.max(x_ice))
        self.assertEqual(0, np.min(x_ice))
        self.assertTrue(np.all(x_ice[:, 0, 0] == 0))
        self.assertTrue(np.all(x_ice[:, -1, 0] == 1))

        for i in range(num_instances):  # x should be ordered in x_0
            self.assertTrue(np.all(np.diff(x_ice[i, :, 0]) > 0))

    def test_create_ice_1D_f_2D(self):
        f = Square.for_n_dimensions(2)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs)
        selected_hyperparameter = cs.get_hyperparameter("x1")

        bo.sample(10)
        num_grid_points = 20
        n_samples = 1000
        ice = ICE.from_random_points(bo.surrogate_model, selected_hyperparameter,
                                     num_samples=n_samples, num_grid_points_per_axis=num_grid_points)

        x_ice = ice.x_ice
        y_ice = ice.y_ice
        variances = ice.y_variances
        grid_points = ice.grid_points

        self.assertEqual(n_samples, x_ice.shape[0])
        self.assertEqual(num_grid_points, x_ice.shape[1])
        self.assertEqual(2, x_ice.shape[2])  # Num dimensions f
        self.assertEqual(num_grid_points, grid_points.shape[0])
        self.assertEqual(1, grid_points.shape[1])  # Num selected hp

        for i in range(num_grid_points):  # x_0 should be the same across all instances
            self.assertTrue(np.all(x_ice[:, i, 0] == x_ice[0, i, 0]))

        for i in range(n_samples):  # x_1 should be the same for all x_0
            self.assertTrue(np.all(np.diff(x_ice[i, :, 1]) == 0))

    def test_create_ice_centered(self):
        f = Square.for_n_dimensions(2)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs)
        selected_hp = cs.get_hyperparameter("x1")

        bo.sample(10)
        ice = ICE.from_random_points(bo.surrogate_model, selected_hp)
        ice.centered = True
        y_ice = ice.y_ice

        self.assertTrue(np.all(y_ice[:, 0] == 0))

    def test_ice_curve_configspace(self):
        f = StyblinskiTang.for_n_dimensions(2)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs)
        selected_hp = cs.get_hyperparameter("x1")

        bo.sample(10)
        ice = ICE.from_random_points(bo.surrogate_model, selected_hp)
        ice_curve = ice[0]
        reduced_cs = ice_curve.implied_config_space
        x1 = reduced_cs.get_hyperparameter("x1")
        x2 = reduced_cs.get_hyperparameter("x2")

        self.assertEqual(selected_hp.lower, x1.lower)
        self.assertEqual(selected_hp.upper, x1.upper)
        self.assertEqual(selected_hp.log, x1.log)
        self.assertIsInstance(x2, CSH.Constant)

    def test_ice_2D_f_3D(self):
        f = Square.for_n_dimensions(3)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs)
        selected_hyperparameter = ["x1", "x2"]

        bo.sample(15)
        num_grid_points_per_axis = 20
        n_samples = 1000
        ice = ICE.from_random_points(
            bo.surrogate_model,
            selected_hyperparameter,
            num_samples=n_samples,
            num_grid_points_per_axis=num_grid_points_per_axis
        )

        x_ice = ice.x_ice
        y_ice = ice.y_ice
        variances = ice.y_variances
        grid_points = ice.grid_points

        self.assertEqual(2, grid_points.shape[1])  # Num selected hp
        self.assertEqual(num_grid_points_per_axis * num_grid_points_per_axis, grid_points.shape[0])
        self.assertEqual(len(grid_points), len(np.unique(grid_points, axis=1)))
        self.assertEqual(n_samples, x_ice.shape[0])
        self.assertEqual(num_grid_points_per_axis * num_grid_points_per_axis, x_ice.shape[1])
        self.assertEqual(3, x_ice.shape[2])  # Num dimensions f

    def test_plot_ice_1D(self):
        self.initialize_figure()

        f = StyblinskiTang.for_n_dimensions(2, seed=42)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs, seed=42)
        selected_hyperparameter = "x1"

        bo.sample(20)
        ice = ICE.from_random_points(
            bo.surrogate_model,
            selected_hyperparameter,
        )

        ice.plot()

    def test_plot_ice_1D_centered(self):
        self.initialize_figure()

        f = StyblinskiTang.for_n_dimensions(2, seed=42)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs, seed=42)
        selected_hyperparameter = "x1"

        bo.sample(20)
        ice = ICE.from_random_points(
            bo.surrogate_model,
            selected_hyperparameter,
        )
        ice.centered = True

        ice.plot()
