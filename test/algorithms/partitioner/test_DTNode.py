from unittest import TestCase

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.blackbox_functions import synthetic_functions, config_space_nd
from src.blackbox_functions.synthetic_functions import Square
from src.sampler.acquisition_function import LowerConfidenceBound
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.surrogate_models import GaussianProcessSurrogate


class TestDTNode(TestCase):
    def setUp(self) -> None:
        # Paper configurations
        f = Square(2)
        bo_sampling_points = 10  # [80, 150, 250]

        # Static paper configurations (not changed throughout the paper)
        self.cs = f.config_space
        self.selected_hyperparameter = self.cs.get_hyperparameter("x1")

        self.surrogate_model = GaussianProcessSurrogate(self.cs)
        bo = BayesianOptimizationSampler(f, self.cs,
                                         surrogate_model=self.surrogate_model,
                                         acq_class=LowerConfidenceBound,
                                         initial_points=2)
        bo.sample(bo_sampling_points)
        ice = ICE(bo.surrogate_model, self.selected_hyperparameter)

        self.partitioner = DTPartitioner.from_ICE(ice)
        self.regions = self.partitioner.partition(max_depth=2)

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

    def test_implied_cs(self):
        root = self.partitioner.root
        root_cs = root.region.implied_config_space(seed=0)

        # root cs should be the same as the original
        for hp in root_cs.get_hyperparameters():
            original_hp = self.cs.get_hyperparameter(hp.name)
            self.assertEqual(hp.upper, original_hp.upper)
            self.assertEqual(hp.lower, original_hp.lower)

        # cs of every leaf node should be different to original
        for leaf in self.partitioner.leaves:
            leaf_cs = leaf.region.implied_config_space(seed=0)
            is_different = False
            for hp in leaf_cs.get_hyperparameters():
                original_hp = self.cs.get_hyperparameter(hp.name)
                if original_hp.lower != hp.lower or original_hp.upper != hp.upper:
                    is_different = True
            self.assertTrue(is_different)
