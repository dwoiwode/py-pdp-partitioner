import unittest

import matplotlib.pyplot as plt

from src.demo_data import blackbox_functions
from src.demo_data.config_spaces import config_space_nd
from src.optimizer import BayesianOptimization, LowerConfidenceBound
from src.partitioner import DecisionTreePartitioner, DTNode
from src.pdp import PDP
from src.plotting import plot_pdp


class TestPaperEvaluations(unittest.TestCase):
    def test_styblinski_tang_8d(self):
        # Paper configurations
        bo_sampling_points = 250  # [80, 150, 250]
        dimensions = 8  # [3,5,8]
        f = blackbox_functions.styblinski_tang_8D

        # Static paper configurations (not changed throughout the paper)
        cs = config_space_nd(dimensions)
        selected_hyperparameter = cs.get_hyperparameters()[0]
        n_samples = 1000
        n_grid_points = 20

        bo = BayesianOptimization(f, cs,
                                  # surrogate_model=surrogate_model,
                                  acq_class=LowerConfidenceBound,
                                  initial_points=4*dimensions)
        bo.optimize(bo_sampling_points - bo.initial_points)
        self.assertEqual(bo_sampling_points, len(bo.y_list))
        incumbent_config, incumbent_score = bo.incumbent
        pdp = PDP(bo.surrogate_model, cs)

        x_ice, y_ice, variances = pdp.calculate_ice(selected_hyperparameter,
                                                    n_grid_points=n_grid_points,
                                                    n_samples=n_samples,
                                                    )
        partitioner = DecisionTreePartitioner(0, x_ice, variances)
        partition_indices, partition_means = partitioner.partition(2)

        correct_leaf = None
        for leaf in partitioner.leaves:
            if incumbent_config in leaf:
                correct_leaf = leaf
                break

        self.assertIsNotNone(correct_leaf)
        self.assertIsInstance(correct_leaf, DTNode)

        filtered_x_ice, filtered_y_ice, filtered_variances = correct_leaf.filter_pdp(x_ice, y_ice, variances)

        plot_pdp(filtered_x_ice, filtered_y_ice, 0)
        plt.show()





