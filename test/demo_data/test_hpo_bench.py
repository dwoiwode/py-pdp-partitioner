from src.algorithms.ice import ICE
from src.algorithms.pdp import PDP
from src.demo_data.hpo_bench import get_SVMBenchmarkMF
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.surrogate_models import GaussianProcessSurrogate
from test import PlottableTest


class TestHPOBench(PlottableTest):
    def test_svm_task_2079(self):
        """
        Took ~3 min for me (dwoiwode)
        """
        seed = 0
        cs, f = get_SVMBenchmarkMF(2079, seed=seed)
        # Paper configurations
        bo_sampling_points = 80  # [80, 160, 240]

        # Static paper configurations (not changed throughout the paper)
        selected_hyperparameter = cs.get_hyperparameter("C")

        # Sampler
        sampler = BayesianOptimizationSampler(f, cs, seed=seed)
        for i in range(3):
            self.initialize_figure()
            sampler.sample(bo_sampling_points)
            sampler.plot(x_hyperparameters=selected_hyperparameter)

            # Surrogate model
            surrogate_model = GaussianProcessSurrogate(cs, seed=seed)
            surrogate_model.fit(sampler.X, sampler.y)
            surrogate_model.plot(x_hyperparameters=selected_hyperparameter)

            # ICE
            ice = ICE(surrogate_model, selected_hyperparameter, seed=seed)
            ice.plot(color="orange")

            # PDP
            pdp = PDP.from_ICE(ice)
            pdp.plot("black", "grey", with_confidence=True)

            # Partitioner
            # dt_partitioner = DTPartitioner(surrogate_model, selected_hyperparamter)

            # Finish plot
            self.save_fig()


