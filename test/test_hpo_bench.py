from unittest import TestCase

from matplotlib import pyplot as plt

from src.algorithms.ice import ICE
from src.demo_data.hpo_bench import get_SVMBenchmarkMF
from src.sampler import BayesianOptimizationSampler, LowerConfidenceBound, RandomSampler
from src.algorithms.pdp import PDP
from src.surrogate_models import GaussianProcessSurrogate


class TestHPOBench(TestCase):
    def test_svm_task_2079(self):
        """
        Took ~3 min for me (dwoiwode)
        """
        cs, f = get_SVMBenchmarkMF(2079)
        # Paper configurations
        bo_sampling_points = 250  # [80, 150, 250]
        dimensions = len(cs.get_hyperparameters())

        # Static paper configurations (not changed throughout the paper)
        selected_hyperparameter = cs.get_hyperparameters()[0]
        n_samples = 1000
        n_grid_points = 20

        seed = 0

        # Sampler
        sampler = BayesianOptimizationSampler(f, cs, seed=seed)
        sampler.sample(bo_sampling_points)
        sampler.plot(x_hyperparameters=selected_hyperparameter)

        # Surrogate model
        surrogate_model = GaussianProcessSurrogate(cs)
        surrogate_model.fit(sampler.X, sampler.y)
        surrogate_model.plot(x_hyperparameters=selected_hyperparameter)

        # ICE
        ice = ICE(surrogate_model, selected_hyperparameter)
        ice.plot(color="orange")

        # PDP
        pdp = PDP.from_ICE(ice)
        pdp.plot("black", "grey", with_confidence=True)

        # Partitioner
        # dt_partitioner = DTPartitioner(surrogate_model, selected_hyperparamter)

        # Finish plot
        plt.legend()
        plt.tight_layout()
        plt.show()


