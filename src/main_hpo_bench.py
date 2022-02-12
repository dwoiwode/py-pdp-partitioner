import warnings
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning

from src.algorithms.ice import ICE
from src.algorithms.pdp import PDP
from src.blackbox_functions.hpo_bench import get_SVMBenchmarkMF
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate

warnings.filterwarnings("ignore", category=ConvergenceWarning)

(Path(__file__).parent.parent / "plots").mkdir(parents=True, exist_ok=True)
plot_folder = Path(__file__).parent.parent / "plots" / "main_hpo_bench"
plot_folder.mkdir(parents=True, exist_ok=True)

def svm_2079():
    seed = 0
    cs, f = get_SVMBenchmarkMF(2079, seed=seed)
    bo_sampling_points = 80
    initial_points = 12

    sampler = BayesianOptimizationSampler(f, cs, seed=seed, acq_class_kwargs={'tau': 1})
    sampler.sample(bo_sampling_points + initial_points)

    surrogate_model = GaussianProcessSurrogate(cs, seed=seed)
    surrogate_model.fit(sampler.X, sampler.y)

    # plot sampler and surrogate
    plt.figure(figsize=(10, 10))
    surrogate_model.plot_means()
    sampler.plot()
    plt.savefig(plot_folder / f'2079_base_Sampler_Mean')
    plt.show()

    plt.figure(figsize=(10, 10))
    surrogate_model.plot_confidences()
    sampler.plot()
    plt.savefig(plot_folder / f'2079_base_Sampler_Confidence')
    plt.show()

    for selected_hp in cs.get_hyperparameters():
        plt.figure(figsize=(10, 10))

        ice = ICE.from_random_points(surrogate_model, selected_hp)
        pdp = PDP.from_ICE(ice, seed)
        pdp.plot_values()
        pdp.plot_confidences()

        plt.savefig(plot_folder / f'2079_base_{selected_hp.name}')
        plt.show()


if __name__ == '__main__':
    svm_2079()
