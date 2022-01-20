from unittest import TestCase

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from src.algorithms.ice import ICE
from src.demo_data import blackbox_functions
from src.demo_data.config_spaces import config_space_nd
from src.sampler import LowerConfidenceBound, BayesianOptimizationSampler
from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.algorithms.pdp import PDP
from src.surrogate_models import SkLearnPipelineSurrogate, GaussianProcessSurrogate


class TestDTNode(TestCase):
    def setUp(self) -> None:
        # Paper configurations
        bo_sampling_points = 10  # [80, 150, 250]
        dimensions = 2
        f = blackbox_functions.square_2D

        # Static paper configurations (not changed throughout the paper)
        self.cs = config_space_nd(dimensions)
        self.selected_hyperparameter = self.cs.get_hyperparameter("x1")

        self.surrogate_model = GaussianProcessSurrogate(self.cs)
        bo = BayesianOptimizationSampler(f, self.cs,
                                         surrogate_model=self.surrogate_model,
                                         acq_class=LowerConfidenceBound,
                                         initial_points=2)
        bo.sample(bo_sampling_points)
        ice = ICE(bo.surrogate_model, self.selected_hyperparameter)

        self.partitioner = DTPartitioner.from_ICE(ice)
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

