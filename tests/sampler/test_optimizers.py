import time

from pyPDP.blackbox_functions.synthetic_functions import Square, NegativeSquare
from pyPDP.sampler.acquisition_function import ProbabilityOfImprovement, LowerConfidenceBound
from pyPDP.sampler.bayesian_optimization import BayesianOptimizationSampler
from pyPDP.utils.plotting import plot_function
from tests import PlottableTest


class TestBayesianSampler(PlottableTest):
    def test_initial_sampling(self):
        f = Square.for_n_dimensions(1)
        cs = f.config_space
        initial_points = 2
        bo = BayesianOptimizationSampler(obj_func=f, config_space=cs,
                                         initial_points=initial_points)
        bo._sample_initial_points()

        self.assertEqual(len(bo.y_list), initial_points)
        self.assertEqual(len(bo.config_list), initial_points)

    def test_find_max_optimum(self):
        self.initialize_figure()
        f = NegativeSquare.for_n_dimensions(1)
        cs = f.config_space

        initial_points = 20
        bo = BayesianOptimizationSampler(obj_func=f, config_space=cs, initial_points=initial_points,
                                         minimize_objective=False)
        bo.sample(50)
        best_config, best_val = bo.incumbent

        # Plot
        plot_function(f, cs)
        bo.plot()
        bo.surrogate_model.plot_means()
        bo.surrogate_model.plot_confidences()

        # Check values
        self.assertAlmostEqual(best_val, 1, delta=1e-3)

    def test_find_min_optimum(self):
        self.initialize_figure()
        seed = 42
        f = Square.for_n_dimensions(1, seed=seed)
        cs = f.config_space

        initial_points = 1
        bo = BayesianOptimizationSampler(obj_func=f, config_space=cs, initial_points=initial_points, seed=seed)
        bo.sample(50)

        best_config, best_val = bo.incumbent

        # Plot
        plot_function(f, cs)
        bo.plot()
        bo.surrogate_model.plot_means()
        bo.surrogate_model.plot_confidences()

        # Check values
        self.assertAlmostEqual(best_val, 0, delta=1e-3)

    def test_use_pi(self):
        self.initialize_figure()
        f = Square.for_n_dimensions(1, seed=0)
        cs = f.config_space

        initial_points = 1
        bo = BayesianOptimizationSampler(obj_func=f,
                                         config_space=cs,
                                         initial_points=initial_points,
                                         acq_class=ProbabilityOfImprovement,
                                         acq_class_kwargs={"eps": 0.01},
                                         seed=0)
        bo.sample(50)

        best_config, best_val = bo.incumbent

        # Plot
        plot_function(f, cs)
        bo.plot()
        bo.surrogate_model.plot_means()
        bo.surrogate_model.plot_confidences()

        # Check values
        print(best_val)
        self.assertAlmostEqual(best_val, 0, delta=1e-3)

    def test_cache_working(self):
        f = Square.for_n_dimensions(1, seed=0)
        cs = f.config_space
        seed = 42
        # First try with seed
        bo = BayesianOptimizationSampler(
            obj_func=f,
            config_space=cs,
            initial_points=1,
            acq_class=ProbabilityOfImprovement,
            acq_class_kwargs={"eps": 0.01},
            seed=seed
        )
        bo.clear_cache()
        (bo.CACHE_DIR / f"{bo.hash}.json").unlink(missing_ok=True)

        t1_1 = time.perf_counter()
        bo.sample(50)
        t1_2 = time.perf_counter()
        del bo  # Trigger saving cache

        # Retry with same seed
        bo2 = BayesianOptimizationSampler(
            obj_func=f,
            config_space=cs,
            initial_points=1,
            acq_class=ProbabilityOfImprovement,
            acq_class_kwargs={"eps": 0.01},
            seed=seed
        )
        t2_1 = time.perf_counter()
        bo2.sample(50)
        t2_2 = time.perf_counter()

        t_dif_1 = t1_2 - t1_1
        t_dif_2 = t2_2 - t2_1
        print(f"Sampling first time with seed {seed} took {t_dif_1}")
        print(f"Sampling second time with seed {seed} took {t_dif_2} ({t_dif_2 / t_dif_1 * 100}%)")

        self.assertGreater(t_dif_1, 1)
        self.assertGreater(0.5, t_dif_2)  # If values are loaded from cache, it is pretty fast

        self.assertGreater(t_dif_1 / 2, t_dif_2)

    def test_cache_sample_more(self):
        f = Square.for_n_dimensions(1, seed=0)
        cs = f.config_space
        seed = 42
        # First try with seed
        bo = BayesianOptimizationSampler(
            obj_func=f,
            config_space=cs,
            initial_points=1,
            acq_class=ProbabilityOfImprovement,
            acq_class_kwargs={"eps": 0.01},
            seed=seed
        )
        bo.clear_cache()
        (bo.CACHE_DIR / f"{bo.hash}.json").unlink(missing_ok=True)

        t1_1 = time.perf_counter()
        bo.sample(20)
        t1_2 = time.perf_counter()
        del bo  # Trigger saving cache

        # Retry with same seed
        bo2 = BayesianOptimizationSampler(
            obj_func=f,
            config_space=cs,
            initial_points=1,
            acq_class=ProbabilityOfImprovement,
            acq_class_kwargs={"eps": 0.01},
            seed=seed
        )
        t2_1 = time.perf_counter()
        bo2.sample(21)
        t2_2 = time.perf_counter()

        t_dif_1 = t1_2 - t1_1
        t_dif_2 = t2_2 - t2_1
        print(f"Sampling first time with seed {seed} took {t_dif_1}")
        print(f"Sampling second time with seed {seed} took {t_dif_2}")

        self.assertGreater(t_dif_2, t_dif_1)
        self.assertNotEqual(bo2.y_list[-1], bo2.y_list[0])  # Last sample should not be same as first sample

    def test_cache_different_acq_class(self):
        f = Square.for_n_dimensions(1, seed=0)
        cs = f.config_space
        seed = 42
        # First try with seed
        bo = BayesianOptimizationSampler(
            obj_func=f,
            config_space=cs,
            initial_points=1,
            acq_class=ProbabilityOfImprovement,
            acq_class_kwargs={"eps": 0.01},
            seed=seed
        )
        bo.clear_cache()
        (bo.CACHE_DIR / f"{bo.hash}.json").unlink(missing_ok=True)

        t1_1 = time.perf_counter()
        bo.sample(20)
        t1_2 = time.perf_counter()
        del bo  # Trigger saving cache

        # Retry with same seed
        bo2 = BayesianOptimizationSampler(
            obj_func=f,
            config_space=cs,
            initial_points=1,
            acq_class=LowerConfidenceBound,
            acq_class_kwargs={"tau": 0.1},
            seed=seed
        )
        t2_1 = time.perf_counter()
        bo2.sample(20)
        t2_2 = time.perf_counter()

        t_dif_1 = t1_2 - t1_1
        t_dif_2 = t2_2 - t2_1
        print(f"Sampling first time with seed {seed} took {t_dif_1}")
        print(f"Sampling second time with seed {seed} took {t_dif_2}")

        self.assertGreater(t_dif_1, 0.1)  # If values are loaded from cache, it is pretty fast
        self.assertGreater(t_dif_2, 0.1)  # If values are loaded from cache, it is pretty fast

    def test_sample_1_init(self):
        f = Square.for_n_dimensions(5, seed=0)
        cs = f.config_space
        # First try with seed
        bo = BayesianOptimizationSampler(
            obj_func=f,
            config_space=cs,
            initial_points=10,
        )
        bo.clear_cache()

        bo.sample(1)
        bo.sample(1)
        bo.sample(1)

        self.assertEqual(3, len(bo))

    def test_sample_1_after_init(self):
        f = Square.for_n_dimensions(5, seed=0)
        cs = f.config_space
        # First try with seed
        bo = BayesianOptimizationSampler(
            obj_func=f,
            config_space=cs,
            initial_points=5,
        )

        bo.sample(5)
        bo.sample(1)
        bo.sample(1)
        self.assertEqual(7, len(bo))
