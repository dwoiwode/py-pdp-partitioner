from src.demo_data.blackbox_functions import square, neg_square
from src.demo_data.config_spaces import config_space_nd
from src.sampler.acquisition_function import ProbabilityOfImprovement
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.utils.plotting import plot_function
from test import PlottableTest


class TestBayesianSampler(PlottableTest):
    def test_initial_sampling(self):
        initial_points = 2
        bo = BayesianOptimizationSampler(obj_func=square, config_space=config_space_nd(1),
                                         initial_points=initial_points)
        bo._sample_initial_points()

        self.assertEqual(len(bo.y_list), initial_points)
        self.assertEqual(len(bo.config_list), initial_points)

    def test_find_max_optimum(self):
        self.initialize_figure()
        cs = config_space_nd(1)

        initial_points = 20
        bo = BayesianOptimizationSampler(obj_func=neg_square, config_space=cs, initial_points=initial_points,
                                         minimize_objective=False)
        bo.sample(50)
        best_config, best_val = bo.incumbent

        # Plot
        plot_function(neg_square, cs)
        bo.plot()
        bo.surrogate_model.plot()

        # Check values
        self.assertAlmostEqual(best_val, 1, delta=1e-3)

    def test_find_min_optimum(self):
        self.initialize_figure()
        cs = config_space_nd(1)

        initial_points = 1
        bo = BayesianOptimizationSampler(obj_func=square, config_space=cs, initial_points=initial_points)
        bo.sample(50)

        best_config, best_val = bo.incumbent

        # Plot
        plot_function(square, cs)
        bo.plot()
        bo.surrogate_model.plot()

        # Check values
        self.assertAlmostEqual(best_val, 0, delta=1e-3)

    def test_use_pi(self):
        self.initialize_figure()
        cs = config_space_nd(1)

        initial_points = 1
        bo = BayesianOptimizationSampler(obj_func=square,
                                         config_space=cs,
                                         initial_points=initial_points,
                                         acq_class=ProbabilityOfImprovement,
                                         acq_class_kwargs={"eps": 0.01})
        bo.sample(50)

        best_config, best_val = bo.incumbent

        # Plot
        plot_function(square, cs)
        bo.plot()
        bo.surrogate_model.plot()

        # Check values
        self.assertAlmostEqual(best_val, 0, delta=1e-3)
