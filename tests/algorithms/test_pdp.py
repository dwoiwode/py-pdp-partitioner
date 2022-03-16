import numpy as np

from pyPDP.algorithms.pdp import PDP
from pyPDP.blackbox_functions.synthetic_functions import Square, StyblinskiTang
from pyPDP.sampler.bayesian_optimization import BayesianOptimizationSampler
from pyPDP.utils.advanced_plots import plot_2D_ICE_Curve_with_confidence
from tests import PlottableTest


class TestPDP(PlottableTest):
    def test_calculate_pdp_1D(self):
        f = Square.for_n_dimensions(1)
        cs = f.config_space
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(f, config_space=cs)
        bo.sample(10)

        pdp = PDP.from_random_points(bo.surrogate_model, selected_hp)

        self.assertTrue(pdp.x_pdp.shape[0] == pdp.y_pdp.shape[0])
        self.assertTrue(np.all(np.diff(pdp.x_pdp) > 0))

    def test_create_pdp_2D(self):
        f = Square.for_n_dimensions(2)
        cs = f.config_space
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(f, config_space=cs)
        bo.sample(10)

        pdp = PDP.from_random_points(bo.surrogate_model, selected_hp)

        self.assertTrue(pdp.x_pdp.shape[0] == pdp.y_pdp.shape[0])
        self.assertTrue(pdp.x_pdp.shape[1] == 2)
        self.assertTrue(np.all(np.diff(pdp.x_pdp[:, 1]) == 0))  # value for feature not in x_s should be the same

    def test_create_pdp_centered(self):
        f = Square.for_n_dimensions(2)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs)

        selected_hp = cs.get_hyperparameter("x1")
        bo.sample(10)

        pdp = PDP.from_random_points(bo.surrogate_model, selected_hp)
        pdp.ice.centered = True

        self.assertTrue(pdp.y_pdp[0] == 0)

    def test_plot_pdp_1D(self):
        self.initialize_figure()

        f = StyblinskiTang.for_n_dimensions(3, seed=42)
        cs = f.config_space
        bo = BayesianOptimizationSampler(f, config_space=cs, seed=42)
        selected_hyperparameter = "x1"

        bo.sample(40)
        pdp = PDP.from_random_points(
            bo.surrogate_model,
            selected_hyperparameter,
        )
        pdp.plot_values()
        pdp.plot_confidences()

    def test_plot_pdp_2D_normal(self):
        self.initialize_figure()

        # Configuration
        f = StyblinskiTang.for_n_dimensions(3, seed=42)
        cs = f.config_space
        selected_hyperparameter = "x1", "x2"

        # Sampler
        bo = BayesianOptimizationSampler(f, config_space=cs, seed=42, initial_points=3 * 4)
        bo.sample(80 + bo.initial_points)

        # PDP
        pdp = PDP.from_random_points(
            bo.surrogate_model,
            selected_hyperparameter,
            num_grid_points_per_axis=20,
            seed=42
        )

        fig, (ax_mean, ax_conf) = plot_2D_ICE_Curve_with_confidence(pdp.as_ice_curve, fig=self.fig)
        bo.plot(ax=ax_mean, x_hyperparameters=selected_hyperparameter)
        bo.plot(ax=ax_conf, x_hyperparameters=selected_hyperparameter)

    def test_plot_pdp_2D_aoversampling(self):
        # "a" in name is there to be before "normal" in alphabet -> cache useable
        self.initialize_figure()

        # Configuration
        f = StyblinskiTang.for_n_dimensions(3, seed=42)
        cs = f.config_space
        selected_hyperparameter = "x1", "x2"

        # Sampler
        bo = BayesianOptimizationSampler(f, config_space=cs, seed=42, initial_points=3 * 4)
        bo.sample(300 + bo.initial_points)

        # PDP
        pdp = PDP.from_random_points(
            bo.surrogate_model,
            selected_hyperparameter,
            num_grid_points_per_axis=20,
            seed=42
        )

        fig, (ax_mean, ax_conf) = plot_2D_ICE_Curve_with_confidence(pdp.as_ice_curve, fig=self.fig)
        bo.plot(ax=ax_mean, x_hyperparameters=selected_hyperparameter)
        bo.plot(ax=ax_conf, x_hyperparameters=selected_hyperparameter)
