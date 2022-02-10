from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.sampler.random_sampler import RandomSampler
from src.utils.plotting import plot_function
from test import PlottableTest


class TestSampler(PlottableTest):
    def test_mmd_random(self):
        f = StyblinskiTang.for_n_dimensions(2, seed=42)
        cs = f.config_space
        n = 550  # Sampler
        m = 550  # Uniform

        # Random
        random_sampler = RandomSampler(f, cs, seed=42)
        random_sampler.sample(n)
        mmd_random = random_sampler.maximum_mean_discrepancy(m=m)
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

        for tau in (0.1, 2, 5):
            # Bayesian
            random_sampler = BayesianOptimizationSampler(f, cs,
                                                         initial_points=f.ndim * 4,
                                                         acq_class_kwargs={"tau": tau})
            random_sampler.sample(n)
            mmd_bayesian = random_sampler.maximum_mean_discrepancy(m=m)
            print(f"Tau: {tau}, MMD: {mmd_bayesian}")