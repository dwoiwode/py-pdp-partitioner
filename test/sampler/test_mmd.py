from matplotlib import pyplot as plt

from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.sampler.random_sampler import RandomSampler
from src.utils.plotting import plot_function
from test import PlottableTest


class TestSampler(PlottableTest):
    def test_mmd_random(self):
        seed = 0
        f = StyblinskiTang.for_n_dimensions(2, seed=seed)
        cs = f.config_space
        n = 2000  # Sampler
        m = 2000  # Uniform

        # Random
        random_sampler = RandomSampler(f, cs, seed=seed)
        random_sampler.sample(n)
        mmd_random = random_sampler.maximum_mean_discrepancy(m=m, seed=None)
        print("MMD Random:", mmd_random)

        # Plot
        self.initialize_figure()
        plot_function(f, cs, 200)
        random_sampler.plot(color=(1, 0, 0))
        plt.legend()

        # Assertion
        self.assertGreater(0.01, mmd_random)

    def test_mmd_compare(self):
        f = StyblinskiTang.for_n_dimensions(2)
        cs = f.config_space
        n = 150  # Sampler
        m = 150  # Uniform

        # Random
        random_sampler = RandomSampler(f, cs)
        random_sampler.sample(n)
        mmd_random = random_sampler.maximum_mean_discrepancy(m=m)
        print(mmd_random)

        # Bayesian Optimization
        bo = BayesianOptimizationSampler(f, cs)
        bo.sample(n)
        mmd_bo = bo.maximum_mean_discrepancy(m=m)
        print(mmd_bo)

        # Plot
        self.initialize_figure()
        plot_function(f, cs, 200)
        random_sampler.plot(color="blue")
        bo.plot(color="red")
        plt.legend()

        # Assertions
        self.assertGreater(mmd_bo, mmd_random)

    def test_mmd_bayesian(self):
        f = StyblinskiTang.for_n_dimensions(5)
        cs = f.config_space
        n = 150  # Sampler
        m = 150  # Uniform
        seed = 0

        for tau in (0.1, 1, 5):
            # Bayesian
            random_sampler = BayesianOptimizationSampler(f, cs,
                                                         initial_points=f.ndim * 4,
                                                         acq_class_kwargs={"tau": tau},
                                                         seed=seed)
            random_sampler.sample(n)
            mmd_bayesian = random_sampler.maximum_mean_discrepancy(m=m, seed=seed)
            print(f"Tau: {tau}, MMD: {mmd_bayesian}")
