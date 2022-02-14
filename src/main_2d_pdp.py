from pathlib import Path

from matplotlib import pyplot as plt

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from src.algorithms.pdp import PDP
from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from src.utils.advanced_plots import plot_2D_ICE_Curve_with_confidence

folder = Path(__file__).parent.parent / "plots" / "main_2d"
folder.mkdir(parents=True, exist_ok=True)


def styblinski_tang_3d():
    f_folder = folder / 'syblinski_3d'
    f_folder.mkdir(parents=True, exist_ok=True)

    seed = 0
    tau = 0.1
    n_dim = 3
    n_samples = 80
    n_initial_samples = n_dim * 4
    selected_hp = ['x1', 'x2']

    f = StyblinskiTang.for_n_dimensions(n_dim, seed=seed)
    sampler = BayesianOptimizationSampler(f, f.config_space, acq_class_kwargs={"tau": tau},
                                          initial_points=n_initial_samples, seed=seed)
    sampler.sample(n_samples + n_initial_samples)

    surrogate = GaussianProcessSurrogate(f.config_space, seed=seed)
    surrogate.fit(sampler.X, sampler.y)

    ice = ICE.from_random_points(surrogate, selected_hp, seed=seed)
    pdp = PDP.from_ICE(ice, seed=seed)

    plt.figure(figsize=(10, 10))
    pdp.plot_values()
    sampler.plot(x_hyperparameters=selected_hp)
    plt.title('Mean values of Surrogate and sampled points')
    plt.savefig(f_folder / 'mean_and_sampler')
    plt.show()

    plt.figure(figsize=(10, 10))
    pdp.plot_confidences()
    sampler.plot(x_hyperparameters=selected_hp)
    plt.title('Confidence values of Surrogate and sampled points')
    plt.savefig(f_folder / 'mean_and_sampler')
    plt.show()

    partitioner = DecisionTreePartitioner.from_ICE(ice)
    partitioner.partition(max_depth=1)
    region = partitioner.get_incumbent_region(sampler.incumbent[0])
    region_pdp = region.pdp_as_ice_curve

    plt.figure(figsize=(10, 10))
    plot_2D_ICE_Curve_with_confidence(region_pdp)
    plt.savefig(f_folder / 'region_pdp')
    plt.show()


if __name__ == '__main__':
    styblinski_tang_3d()
