from unittest import TestCase

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from src.demo_data import blackbox_functions
from src.demo_data.config_spaces import config_space_nd
from src.optimizer import LowerConfidenceBound, BayesianOptimization
from src.partitioner import DecisionTreePartitioner
from src.pdp import PDP


class TestDTNode(TestCase):
    def setUp(self) -> None:
        # Paper configurations
        bo_sampling_points = 10  # [80, 150, 250]
        dimensions = 2
        f = blackbox_functions.square_2D

        # Static paper configurations (not changed throughout the paper)
        self.cs = config_space_nd(dimensions)
        selected_hyperparameter = self.cs.get_hyperparameters()[0]
        n_samples = 100
        n_grid_points = 10

        surrogate_model = GaussianProcessRegressor(
            kernel=Matern(nu=3 / 2)
        )
        bo = BayesianOptimization(f, self.cs,
                                  surrogate_model=surrogate_model,
                                  acq_class=LowerConfidenceBound,
                                  initial_points=2)
        bo.optimize(bo_sampling_points - bo.initial_points)
        pdp = PDP(bo.surrogate_model, self.cs)

        x_ice, y_ice, variances = pdp.calculate_ice(selected_hyperparameter,
                                                    n_grid_points=n_grid_points,
                                                    n_samples=n_samples)
        self.partitioner = DecisionTreePartitioner(0, x_ice, variances)
        self.partitioner.partition(2)

    def test_contains_root_node(self):
        root = self.partitioner.root
        random_config = self.cs.sample_configuration()
        self.assertTrue(random_config in root)


    def test_contains_leaf_node(self):
        for i in range(1000):
            n_contained_leaves = 0
            config = self.cs.sample_configuration()
            for leaf in self.partitioner.leaves:
                if config in leaf:
                    n_contained_leaves += 1

            self.assertEqual(1, n_contained_leaves)

