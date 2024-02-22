import unittest

from matplotlib import pyplot as plt

from pyPDP.algorithms.ice import ICE
from pyPDP.algorithms.pdp import PDP
from pyPDP.blackbox_functions.hpo_bench import get_SVMBenchmarkMF, get_RFBenchmarkMF, get_NNBenchmarkMF
from pyPDP.sampler.bayesian_optimization import BayesianOptimizationSampler
from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from tests import PlottableTest


class TestHPOBench(PlottableTest):
    @unittest.SkipTest
    def test_svm_task_2079(self):
        """
        Took ~3 min for me (dwoiwode)

        SVM has 2 hyperparameter: C, gamma
        Each parameter is ranged from ~0 -> 1024 in log-space
        """
        seed = 0
        cs, f = get_SVMBenchmarkMF(2079, seed=seed)
        # Paper configurations
        bo_sampling_points = 5  # [80, 160, 240]

        # Sampler
        sampler = BayesianOptimizationSampler(f, cs, seed=seed)
        for i in range(3):
            sampler.sample(bo_sampling_points)

            # Surrogate model
            surrogate_model = GaussianProcessSurrogate(cs, seed=seed)
            surrogate_model.fit(sampler.X, sampler.y)
            for selected_hyperparameter in cs.get_hyperparameters():
                self.initialize_figure()
                self.fig.suptitle(f"HPOBench SVM Task 2079 - {selected_hyperparameter.name} - {len(sampler)} samples")

                # Plot sampler/surrogate
                sampler.plot(x_hyperparameters=selected_hyperparameter)

                # ICE
                ice = ICE.from_random_points(surrogate_model, selected_hyperparameter, seed=seed)
                ice.plot(color="orange")

                # PDP
                pdp = PDP.from_ICE(ice)
                pdp.plot_values("black")
                pdp.plot_confidences("grey")

                # Partitioner
                # dt_partitioner = DecisionTreePartitioner.from_ICE(ice)
                self.save_fig()
                plt.show()

    @unittest.SkipTest
    def test_rf_task_2079(self):
        seed = 0
        cs, f = get_RFBenchmarkMF(2079, seed=seed)
        # Paper configurations
        bo_sampling_points = 5  # [80, 160, 240]

        # Sampler
        sampler = BayesianOptimizationSampler(f, cs, seed=seed)
        for i in range(3):
            sampler.sample(bo_sampling_points)

            # Surrogate model
            surrogate_model = GaussianProcessSurrogate(cs, seed=seed)
            surrogate_model.fit(sampler.X, sampler.y)
            for selected_hyperparameter in cs.get_hyperparameters():
                self.initialize_figure()
                self.fig.suptitle(f"HPOBench RF Task 2079 - {selected_hyperparameter.name} - {len(sampler)} samples")

                # Plot sampler/surrogate
                sampler.plot(x_hyperparameters=selected_hyperparameter)

                # ICE
                ice = ICE.from_random_points(surrogate_model, selected_hyperparameter, seed=seed)
                ice.plot(color="orange")

                # PDP
                pdp = PDP.from_ICE(ice)
                pdp.plot_values("black")
                pdp.plot_confidences("grey")

                # Partitioner
                # dt_partitioner = DecisionTreePartitioner.from_ICE(ice)
                self.save_fig()
                plt.show()

    @unittest.SkipTest
    def test_nn_task_2079(self):
        seed = 0
        cs, f = get_NNBenchmarkMF(2079, seed=seed)
        # Paper configurations
        bo_sampling_points = 5  # [80, 160, 240]

        # Sampler
        sampler = BayesianOptimizationSampler(f, cs, seed=seed)
        for i in range(3):
            sampler.sample(bo_sampling_points)

            # Surrogate model
            surrogate_model = GaussianProcessSurrogate(cs, seed=seed)
            surrogate_model.fit(sampler.X, sampler.y)
            for selected_hyperparameter in cs.get_hyperparameters():
                self.initialize_figure()
                self.fig.suptitle(f"HPOBench NN Task 2079 - {selected_hyperparameter.name} - {len(sampler)} samples")

                # Plot sampler/surrogate
                sampler.plot(x_hyperparameters=selected_hyperparameter)

                # ICE
                ice = ICE.from_random_points(surrogate_model, selected_hyperparameter, seed=seed)
                ice.plot(color="orange")

                # PDP
                pdp = PDP.from_ICE(ice)
                pdp.plot_values("black")
                pdp.plot_confidences("grey")

                # Partitioner
                # dt_partitioner = DecisionTreePartitioner.from_ICE(ice)
                self.save_fig()
                plt.show()
