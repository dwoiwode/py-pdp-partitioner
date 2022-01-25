from matplotlib import pyplot as plt

from src.demo_data.blackbox_functions import styblinski_tang_2D, styblinski_tang_5D
from src.demo_data.config_spaces import config_space_nd
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.sampler.random_sampler import RandomSampler
from src.utils.plotting import plot_function
from test import PlottableTest


class TestSampler(PlottableTest):
    def test_mmd_random(self):
        f = styblinski_tang_2D
        cs = config_space_nd(2)
        n = 250  # Sampler
        m = 500  # Uniform

        # Random
        random_sampler = RandomSampler(f, cs)
        random_sampler.sample(n)
        mmd_random = random_sampler.maximum_mean_discrepancy(m=m)
        print("MMD Random:", mmd_random)

        # Plot
        self.initialize_figure()
        plot_function(f, cs, 200)
        random_sampler.plot(color=(1, 0, 0))
        plt.legend()

        # Assertion
        self.assertAlmostEqual(0, mmd_random, places=3)

    def test_mmd_compare(self):
        f = styblinski_tang_2D
        cs = config_space_nd(2)
        n = 150  # Sampler
        m = 1000  # Uniform

        # Random
        random_sampler = RandomSampler(f, cs)
        random_sampler.sample(n)
        # l2_distance = random_sampler.median_distance_between_points()
        # print("L2_random", l2_distance)
        mmd_random = random_sampler.maximum_mean_discrepancy(m=m)
        print(mmd_random)

        # Bayesian Optimization
        bo = BayesianOptimizationSampler(f, cs)
        bo.sample(n)
        # l2_distance = bo.median_distance_between_points()
        # print("L2_bo", l2_distance)
        mmd_bo = bo.maximum_mean_discrepancy(m=m)
        print(mmd_bo)

        # Plot
        self.initialize_figure()
        plot_function(f, cs, 200)
        random_sampler.plot(color="blue")
        bo.plot(color="red")
        plt.legend()
        self.save_fig()

        # Assertions
        self.assertGreater(mmd_bo, mmd_random)

    def test_mmd_bayesian(self):
        f = styblinski_tang_5D
        d = 5
        cs = config_space_nd(d)
        n = 150  # Sampler
        m = 500  # Uniform

        for tau in (0.1, 2, 5):
            # Bayesian
            random_sampler = BayesianOptimizationSampler(f, cs,
                                                         initial_points=d * 4,
                                                         acq_class_kwargs={"tau": tau})
            random_sampler.sample(n)
            mmd_bayesian = random_sampler.maximum_mean_discrepancy(m=m)
            print(f"Tau: {tau}, MMD: {mmd_bayesian}")
