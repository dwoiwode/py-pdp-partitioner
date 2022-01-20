import unittest

import matplotlib.pyplot as plt

from src.demo_data import blackbox_functions
from src.demo_data.config_spaces import config_space_nd
from src.sampler import BayesianOptimization, LowerConfidenceBound
from src.algorithms.partitioner import DecisionTreePartitioner, DTNode
from src.algorithms.pdp import PDP
from src.plotting import plot_pdp, plot_ice, plot_confidence_lists


class TestPaperEvaluations(unittest.TestCase):
    def test_styblinski_tang_8d(self):
        # Paper configurations
        bo_sampling_points = 80  # [80, 150, 250]
        dimensions = 8  # [3,5,8]
        f = blackbox_functions.styblinski_tang_8D
        # f = blackbox_functions.styblinski_tang_3D

        # Static paper configurations (not changed throughout the paper)
        cs = config_space_nd(dimensions)
        selected_hyperparameter = cs.get_hyperparameters()[0]
        n_samples = 1000
        n_grid_points = 20

        bo = BayesianOptimization(f, cs,
                                  # surrogate_model=surrogate_model,
                                  acq_class=LowerConfidenceBound,
                                  initial_points=4*dimensions)
        bo.sample(bo_sampling_points - bo.initial_points)
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

        filtered_x_pdp, filtered_y_pdp, filtered_variances_pdp = correct_leaf.filter_pdp(x_ice, y_ice, variances)
        filtered_x_ice, filtered_y_ice, filtered_variances_ice = correct_leaf.filter_ice(x_ice, y_ice, variances)

        ax = plot_ice(filtered_x_ice, filtered_y_ice, idx=0, alpha=0.1)
        plot_pdp(filtered_x_pdp, filtered_y_pdp, idx=0, ax=ax, alpha=1)
        plot_confidence_lists(filtered_x_pdp[:, 0], filtered_y_pdp, variances=filtered_variances_pdp, ax=ax)
        plt.show()





