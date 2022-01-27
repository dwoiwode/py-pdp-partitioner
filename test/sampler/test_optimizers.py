from src.blackbox_functions import config_space_nd
from src.blackbox_functions.synthetic_functions import Square, NegativeSquare
from src.sampler.acquisition_function import ProbabilityOfImprovement
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.utils.plotting import plot_function
from test import PlottableTest


class TestBayesianSampler(PlottableTest):
    def test_initial_sampling(self):
        f = Square.for_n_dimensions(1)
        cs = f.config_space
        initial_points = 2
        bo = BayesianOptimizationSampler(obj_func=f, config_space=cs,
                                         initial_points=initial_points)
        bo._sample_initial_points()

        self.assertEqual(len(bo.y_list), initial_points)
        self.assertEqual(len(bo.config_list), initial_points)

    def test_find_max_optimum(self):
        self.initialize_figure()
        f = NegativeSquare.for_n_dimensions(1)
        cs = f.config_space

        initial_points = 20
        bo = BayesianOptimizationSampler(obj_func=f, config_space=cs, initial_points=initial_points,
                                         minimize_objective=False)
        bo.sample(50)
        best_config, best_val = bo.incumbent

        # Plot
        plot_function(f, cs)
        bo.plot()
        bo.surrogate_model.plot()

        # Check values
        self.assertAlmostEqual(best_val, 1, delta=1e-3)

    def test_find_min_optimum(self):
        self.initialize_figure()
        f = Square.for_n_dimensions(1)
        cs = f.config_space

        initial_points = 1
        bo = BayesianOptimizationSampler(obj_func=f, config_space=cs, initial_points=initial_points)
        bo.sample(50)

        best_config, best_val = bo.incumbent

        # Plot
        plot_function(f, cs)
        bo.plot()
        bo.surrogate_model.plot()

        # Check values
        self.assertAlmostEqual(best_val, 0, delta=1e-3)

    def test_use_pi(self):
        self.initialize_figure()
        f = Square.for_n_dimensions(1, seed=0)
        cs = f.config_space

        initial_points = 1
        bo = BayesianOptimizationSampler(obj_func=f,
                                         config_space=cs,
                                         initial_points=initial_points,
                                         acq_class=ProbabilityOfImprovement,
                                         acq_class_kwargs={"eps": 0.01})
        bo.sample(50)

        best_config, best_val = bo.incumbent

        # Plot
        plot_function(f, cs)
        bo.plot()
        bo.surrogate_model.plot()

        # Check values
        print(best_val)
        self.assertAlmostEqual(best_val, 0, delta=1e-4)
