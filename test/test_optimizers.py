import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import matplotlib.pyplot as plt

from src.demo_data.blackbox_functions import square, neg_square
from src.demo_data.config_spaces import config_space_nd
from src.sampler import BayesianOptimization, LowerConfidenceBound
from src.plotting import plot_function, plot_samples, plot_model_confidence
from test.test_plotting import TestPlotting


class TestBayesianOptimizer(TestPlotting):
    def test_initial_sampling(self):
        self.no_plot = True

        initial_points = 2
        bo = BayesianOptimization(obj_func=square, config_space=config_space_nd(1), initial_points=initial_points, eps=0.001)
        bo.sample(0)

        self.assertEqual(len(bo.y_list), initial_points)
        self.assertEqual(len(bo.config_list), initial_points)

    def test_find_max_optimum(self):
        x = CSH.UniformFloatHyperparameter("x1", lower=-1, upper=1)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        initial_points = 20
        bo = BayesianOptimization(obj_func=neg_square, config_space=cs, initial_points=initial_points, eps=0,
                                  minimize_objective=False)
        best_config = bo.sample(50)
        best_val = best_config['x1']

        fig = plt.figure()
        ax = fig.gca()
        plot_function(neg_square, cs, ax=ax)
        plot_samples(bo.config_list, bo.y_list, ax=ax)
        plot_model_confidence(bo, cs, ax=ax)

        self.assertAlmostEqual(best_val, 0, delta=1e-3)

    def test_find_min_optimum(self):
        x = CSH.UniformFloatHyperparameter("x1", lower=-1, upper=1)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        initial_points = 1
        bo = BayesianOptimization(obj_func=square, config_space=cs, initial_points=initial_points, eps=0,
                                  minimize_objective=True)

        best_config = bo.sample(50)
        best_val = best_config['x1']

        fig = plt.figure()
        ax = fig.gca()
        plot_function(square, cs, ax=ax)
        plot_samples(bo.config_list, bo.y_list, ax=ax)
        plot_model_confidence(bo, cs, ax=ax)
        self.assertAlmostEqual(best_val, 0, delta=1e-3)

    def test_use_lcb(self):
        x = CSH.UniformFloatHyperparameter("x1", lower=-1, upper=1)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        initial_points = 1
        bo = BayesianOptimization(obj_func=square, config_space=cs, initial_points=initial_points, eps=0,
                                  minimize_objective=True, acq_class=LowerConfidenceBound)

        best_config = bo.sample(50)
        best_val = best_config['x1']

        fig = plt.figure()
        ax = fig.gca()
        plot_function(square, cs, ax=ax)
        plot_samples(bo.config_list, bo.y_list, ax=ax)
        plot_model_confidence(bo, cs, ax=ax)
        self.assertAlmostEqual(best_val, 0, delta=1e-3)
