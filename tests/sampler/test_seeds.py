from unittest import TestCase

import numpy as np

from pyPDP.algorithms.ice import ICE
from pyPDP.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from pyPDP.algorithms.pdp import PDP
from pyPDP.blackbox_functions.synthetic_functions import StyblinskiTang
from pyPDP.sampler.bayesian_optimization import BayesianOptimizationSampler
from pyPDP.sampler.random_sampler import RandomSampler
from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate


class TestSeeds(TestCase):
    seed = 2312

    def test_seeds_bo(self):
        """
        Test whether same seed achieves same results everytime and different seeds achieve different results
        """
        n_samples = 50
        # Create functions
        f_seeded_1 = StyblinskiTang.for_n_dimensions(5, seed=self.seed)
        f_seeded_2 = StyblinskiTang.for_n_dimensions(5, seed=self.seed)
        f_not_seeded = StyblinskiTang.for_n_dimensions(5)

        # Create sampler
        bo_sampler_seeded_1 = BayesianOptimizationSampler(f_seeded_1, f_seeded_1.config_space, seed=self.seed)
        bo_sampler_seeded_2 = BayesianOptimizationSampler(f_seeded_2, f_seeded_2.config_space, seed=self.seed)
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
        bo_mmd_seeded_1 = bo_sampler_seeded_1.maximum_mean_discrepancy(20, seed=self.seed)
        bo_mmd_seeded_2 = bo_sampler_seeded_2.maximum_mean_discrepancy(20, seed=self.seed)
        bo_mmd_not_seeded = bo_sampler_not_seeded.maximum_mean_discrepancy(20, seed=None)
        self.assertEqual(bo_mmd_seeded_1,
                         bo_mmd_seeded_2)
        self.assertNotEqual(bo_mmd_seeded_1,
                            bo_mmd_not_seeded)

        # Surrogate
        # All models will have the same samples to fit on
        surrogate_seeded_1 = GaussianProcessSurrogate(f_seeded_1.config_space, seed=self.seed)
        surrogate_seeded_2 = GaussianProcessSurrogate(f_seeded_2.config_space, seed=self.seed)
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
        ice_seeded_1 = ICE.from_random_points(surrogate_seeded_1, selected_hp, seed=self.seed)
        ice_seeded_2 = ICE.from_random_points(surrogate_seeded_2, selected_hp, seed=self.seed)
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
        best_config_seeded_1 = best_region_seeded_1.implied_config_space(self.seed)
        best_config_seeded_2 = best_region_seeded_2.implied_config_space(self.seed)
        self.assertEqual(best_config_seeded_1, best_config_seeded_2)

        # Check metrics
        self.assertEqual(best_region_seeded_1.loss, best_region_seeded_2.loss)
        self.assertEqual(best_region_seeded_1.mean_confidence, best_region_seeded_2.mean_confidence)
        self.assertEqual(best_region_seeded_1.negative_log_likelihood(f_seeded_1),
                         best_region_seeded_2.negative_log_likelihood(f_seeded_2))

    def test_seed_from_pdp(self):
        GRID_POINTS_PER_AXIS = 10
        num_samples = 50

        f = StyblinskiTang.for_n_dimensions(4)
        sampler = RandomSampler(f, f.config_space)
        sampler.sample(50)

        surrogate_model = GaussianProcessSurrogate(f.config_space)
        surrogate_model.fit(sampler.X, sampler.y)

        selected_hyperparameters = ["x1"]

        # And finally call PDP
        pdp1 = PDP.from_random_points(
            surrogate_model,
            selected_hyperparameter=selected_hyperparameters,
            seed=self.seed,
            num_grid_points_per_axis=GRID_POINTS_PER_AXIS,
            num_samples=num_samples,
        )
        pdp2 = PDP.from_random_points(
            surrogate_model,
            selected_hyperparameter=selected_hyperparameters,
            seed=self.seed,
            num_grid_points_per_axis=GRID_POINTS_PER_AXIS,
            num_samples=num_samples,
        )

        x1 = pdp1.x_pdp.tolist()
        y1 = pdp1.y_pdp.tolist()
        x2 = pdp2.x_pdp.tolist()
        y2 = pdp2.y_pdp.tolist()

        self.assertTrue(np.array_equal(x1, x2))
        self.assertTrue(np.array_equal(y1, y2))
