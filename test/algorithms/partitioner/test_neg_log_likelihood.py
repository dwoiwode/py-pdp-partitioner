import unittest

from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.demo_data.blackbox_functions import styblinski_tang_5D, styblinski_tang_3D, styblinski_tang_3D_int_2D, \
    styblinski_tang_3D_int_1D
from src.demo_data.config_spaces import config_space_nd
from src.sampler.bayesian_optimization import BayesianOptimizationSampler


class TestICE(unittest.TestCase):
    def test_nll(self):
        f = styblinski_tang_3D
        d = 3
        cs = config_space_nd(d)
        n = 80
        tau = 2
        selected_hyperparameter = cs.get_hyperparameter("x1")

        # Bayesian
        random_sampler = BayesianOptimizationSampler(f, cs,
                                                     initial_points=d * 4,
                                                     acq_class_kwargs={"tau": tau})
        random_sampler.sample(n)

        dt_partitioner = DTPartitioner(random_sampler.surrogate_model, selected_hyperparameter)
        leaf_list = dt_partitioner.partition(max_depth=1)
        best_region = dt_partitioner.get_incumbent_region(random_sampler.incumbent[0])

        true_pd = styblinski_tang_3D_int_1D
        nll = best_region.negative_log_likelihood(true_pd_function=true_pd)

        print(nll)



