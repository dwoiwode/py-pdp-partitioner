from unittest import TestCase

import numpy as np

from pyPDP.algorithms.ice import ICE
from pyPDP.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from pyPDP.blackbox_functions.synthetic_functions import StyblinskiTang
from pyPDP.sampler.bayesian_optimization import BayesianOptimizationSampler
from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate


class TestSeeds(TestCase):
    def test_seeds_bo(self):
        """
        Test whether same seed achieves same results everytime and different seeds achieve different results
        """
        seed = 2312
        n_samples = 50
        # Create functions
        f_seeded_1 = StyblinskiTang.for_n_dimensions(5, seed=seed)
        f_seeded_2 = StyblinskiTang.for_n_dimensions(5, seed=seed)
        f_not_seeded = StyblinskiTang.for_n_dimensions(5)

        # Create sampler
        bo_sampler_seeded_1 = BayesianOptimizationSampler(f_seeded_1, f_seeded_1.config_space, seed=seed)
        bo_sampler_seeded_2 = BayesianOptimizationSampler(f_seeded_2, f_seeded_2.config_space, seed=seed)
        bo_sampler_not_seeded = BayesianOptimizationSampler(f_not_seeded, f_not_seeded.config_space)

        # Sample points
        bo_sampler_seeded_1.sample(n_samples)
        bo_sampler_seeded_2.sample(n_samples)
        bo_sampler_not_seeded.sample(n_samples)

        # Check equality of sampler
        # X
        self.assertTrue(np.array_equal(bo_sampler_seeded_1.X, bo_sampler_seeded_2.X))
        self.assertFalse(np.array_equal(bo_sampler_seeded_1.X, bo_sampler_not_seeded.X))

        # Y
        self.assertTrue(np.array_equal(bo_sampler_seeded_1.y, bo_sampler_seeded_2.y))
        self.assertFalse(np.array_equal(bo_sampler_seeded_1.y, bo_sampler_not_seeded.y))

        # Maximum mean discrepancy
        bo_mmd_seeded_1 = bo_sampler_seeded_1.maximum_mean_discrepancy(20, seed=seed)
        bo_mmd_seeded_2 = bo_sampler_seeded_2.maximum_mean_discrepancy(20, seed=seed)
        bo_mmd_not_seeded = bo_sampler_not_seeded.maximum_mean_discrepancy(20, seed=None)
        self.assertEqual(bo_mmd_seeded_1,
                         bo_mmd_seeded_2)
        self.assertNotEqual(bo_mmd_seeded_1,
                            bo_mmd_not_seeded)

        # Surrogate
        # All models will have the same samples to fit on
        surrogate_seeded_1 = GaussianProcessSurrogate(f_seeded_1.config_space, seed=seed)
        surrogate_seeded_2 = GaussianProcessSurrogate(f_seeded_2.config_space, seed=seed)
        surrogate_not_seeded = GaussianProcessSurrogate(f_not_seeded.config_space)

        surrogate_seeded_1.fit(bo_sampler_seeded_1.X, bo_sampler_seeded_1.y)
        surrogate_seeded_2.fit(bo_sampler_seeded_1.X, bo_sampler_seeded_1.y)
        surrogate_not_seeded.fit(bo_sampler_seeded_1.X, bo_sampler_seeded_1.y)

        # Check predictions
        predictions_seeded_1 = surrogate_seeded_1.predict(bo_sampler_seeded_1.X)
        predictions_seeded_2 = surrogate_seeded_2.predict(bo_sampler_seeded_1.X)

        self.assertTrue(np.array_equal(predictions_seeded_1, predictions_seeded_2))
        # Model might be deterministic -> Cannot guarantee that not seeded != seeded

        # ICE
        selected_hp = f_seeded_1.config_space.get_hyperparameter("x1")
        ice_seeded_1 = ICE.from_random_points(surrogate_seeded_1, selected_hp, seed=seed)
        ice_seeded_2 = ICE.from_random_points(surrogate_seeded_2, selected_hp, seed=seed)
        ice_not_seeded = ICE.from_random_points(surrogate_not_seeded, selected_hp)

        # Check equality of Surrogate/ICE
        # X
        self.assertTrue(np.array_equal(ice_seeded_1.x_ice, ice_seeded_2.x_ice),
                        f"{ice_seeded_1.x_ice} != {ice_seeded_2.x_ice}")
        self.assertFalse(np.array_equal(ice_seeded_1.x_ice, ice_not_seeded.x_ice))

        # Y
        self.assertTrue(np.array_equal(ice_seeded_1.y_ice, ice_seeded_2.y_ice))
        self.assertFalse(np.array_equal(ice_seeded_1.y_ice, ice_not_seeded.y_ice))

        # Partitioner
        # DTPartitioner is deterministic. Cannot check for != seed
        dt_seeded_1 = DecisionTreePartitioner.from_ICE(ice_seeded_1)
        dt_seeded_2 = DecisionTreePartitioner.from_ICE(ice_seeded_2)
        dt_seeded_1.partition(3)
        dt_seeded_2.partition(3)
        best_region_seeded_1 = dt_seeded_1.get_incumbent_region(bo_sampler_seeded_1.incumbent_config)
        best_region_seeded_2 = dt_seeded_1.get_incumbent_region(bo_sampler_seeded_2.incumbent_config)

        # Check Split conditions
        for split_seeded_1, split_seeded_2 in zip(best_region_seeded_1.split_conditions,
                                                  best_region_seeded_2.split_conditions):
            self.assertEqual(split_seeded_1, split_seeded_2)

        # Incumbent config space
        best_config_seeded_1 = best_region_seeded_1.implied_config_space(seed)
        best_config_seeded_2 = best_region_seeded_2.implied_config_space(seed)
        self.assertEqual(best_config_seeded_1, best_config_seeded_2)

        # Check metrics
        self.assertEqual(best_region_seeded_1.loss, best_region_seeded_2.loss)
        self.assertEqual(best_region_seeded_1.mean_confidence, best_region_seeded_2.mean_confidence)
        self.assertEqual(best_region_seeded_1.negative_log_likelihood(f_seeded_1),
                         best_region_seeded_2.negative_log_likelihood(f_seeded_2))
