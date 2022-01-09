from unittest import TestCase
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from src.blackbox_functions import ackley, square, neg_square
from src.optimizer import BayesianOptimization
from src.utils import config_to_array, plot_function


class TestBayesianOptimizer(TestCase):
    def test_initial_sampling(self):
        x = CSH.UniformFloatHyperparameter("x", lower=-1, upper=1)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        initial_points = 2
        bo = BayesianOptimization(obj_func=square, config_space=cs, initial_points=initial_points, eps=0.001)
        bo.optimize(0)

        self.assertEqual(len(bo.y_list), initial_points)
        self.assertEqual(len(bo.config_list), initial_points)

    def test_find_max_optimum(self):
        x = CSH.UniformFloatHyperparameter("x", lower=-1, upper=1)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        initial_points = 20
        bo = BayesianOptimization(obj_func=neg_square, config_space=cs, initial_points=initial_points, eps=0,
                                  minimize_objective=False)
        best_config = bo.optimize(500)
        best_val = config_to_array(best_config)[0]

        fig = plot_function(neg_square, cs, config_samples=bo.config_list, model=bo.surrogate_score)
        fig.show()

        self.assertAlmostEqual(best_val, 0, delta=1E-3)

    def test_find_min_optimum(self):
        x = CSH.UniformFloatHyperparameter("x", lower=-1, upper=1)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        initial_points = 1
        bo = BayesianOptimization(obj_func=square, config_space=cs, initial_points=initial_points, eps=0,
                                  minimize_objective=True)

        best_config = bo.optimize(500)
        best_val = config_to_array(best_config)[0]

        fig = plot_function(square, cs, config_samples=bo.config_list, model=bo.surrogate_score)
        fig.show()
        self.assertAlmostEqual(best_val, 0, delta=1E-3)
