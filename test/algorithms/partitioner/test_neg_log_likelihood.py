import unittest

from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler.bayesian_optimization import BayesianOptimizationSampler


class TestNLL(unittest.TestCase):
    def test_nll(self):
        f = StyblinskiTang.for_n_dimensions(3, seed=32)
        cs = f.config_space
        n = 80
        tau = 1
        selected_hyperparameter = cs.get_hyperparameter("x1")

        # Bayesian
        random_sampler = BayesianOptimizationSampler(f, cs,
                                                     initial_points=f.ndim * 4,
                                                     acq_class_kwargs={"tau": tau})
        random_sampler.sample(n)

        dt_partitioner = DTPartitioner.from_random_points(random_sampler.surrogate_model, selected_hyperparameter)
        leaf_list = dt_partitioner.partition(max_depth=1)
        best_region = dt_partitioner.get_incumbent_region(random_sampler.incumbent[0])

        nll = best_region.negative_log_likelihood(true_function=f)

        print(nll)



