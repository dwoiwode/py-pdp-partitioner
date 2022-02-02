import warnings
from typing import Dict, Type

import numpy as np
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning

from src.algorithms.ice import ICECurve
from src.algorithms.pdp import PDP
from src.blackbox_functions import BlackboxFunction, BlackboxFunctionND
from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler import Sampler
from src.sampler.acquisition_function import LowerConfidenceBound
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.sampler.grid_sampler import GridSampler
from src.sampler.random_sampler import RandomSampler
from src.surrogate_models import GaussianProcessSurrogate
from src.utils.plotting import plot_function
from tqdm import tqdm

from src.utils.utils import unscale, get_uniform_distributed_ranges, convert_hyperparameters

warnings.filterwarnings("ignore", category=ConvergenceWarning)
seed = 0


def plot_sampling_bias(f_class: Type[BlackboxFunctionND] = StyblinskiTang,
                       dimensions=2,
                       sampler_factories: Dict[str, Sampler] = None,
                       sampled_points=56,
                       repetitions=10,
                       seed_offset=0, ):
    f = f_class.for_n_dimensions(dimensions)
    cs = f.config_space
    initial_points = 4 * f.ndim
    if sampler_factories is None:
        # Default sampler (from paper)
        sampler_factories = {
            # "High sampling bias": lambda seed: BayesianOptimizationSampler(
            #     f,
            #     cs,
            #     initial_points=initial_points,
            #     acq_class=LowerConfidenceBound,
            #     acq_class_kwargs={"tau": 0.1},
            #     seed=seed
            # ),
            # "Medium Sampling bias": lambda seed: BayesianOptimizationSampler(
            #     f,
            #     cs,
            #     initial_points=initial_points,
            #     acq_class=LowerConfidenceBound,
            #     acq_class_kwargs={"tau": 2},
            #     seed=seed
            # ),
            "Random": lambda seed: RandomSampler(
                f,
                cs,
                seed=seed
            ),
            "Grid": lambda seed: GridSampler(
                f,
                cs,
                seed=seed
            ),
        }

    selected_hyperparameter = ["x1"]

    n = len(sampler_factories)
    fig, axes = plt.subplots(2, n, sharex="all", sharey="row", figsize=(4 * n, 4))

    for (name, sampler_factory), ax3 in zip(sampler_factories.items(), axes.T):
        arr_means = []
        arr_variances = []
        arr_x = []

        for i in tqdm(range(repetitions), desc=f"Sampler: {name}"):
            seed = seed_offset + i
            f = f_class.for_n_dimensions(dimensions, seed=seed)
            cs = f.config_space

            sampler = sampler_factory(seed)
            assert isinstance(sampler, Sampler)
            sampler.sample(sampled_points + initial_points)
            surrogate = GaussianProcessSurrogate(cs, seed=seed)
            surrogate.fit(sampler.X, sampler.y)
            pdp = PDP.from_random_points(surrogate_model=surrogate, selected_hyperparameter=selected_hyperparameter)
            arr_x = pdp.x_pdp
            arr_means.append(pdp.y_pdp)
            arr_variances.append(pdp.y_variances)

        arr_means = np.mean(arr_means, axis=0)
        arr_variances = np.mean(arr_variances, axis=0)

        ax_pdp = ax3[0]
        ax_variances = ax3[1]
        assert isinstance(ax_pdp, plt.Axes)
        assert isinstance(ax_variances, plt.Axes)

        # Plot pdp
        mean_pdp = ICECurve(
            full_config_space=cs,
            selected_hyperparameter=selected_hyperparameter,
            x_ice=arr_x,
            y_ice=arr_means,
            y_variances=arr_variances,
            name=f"Mean PDP {name}"
        )
        mean_pdp.plot(line_color="blue", gradient_color="lightblue", with_confidence=True, ax=ax_pdp)

        f_pd = f.pd_integral(*[hp for hp in cs if hp not in selected_hyperparameter])
        plot_function(f_pd, f_pd.config_space, samples_per_axis=200, ax=ax_pdp)

        # Plot variances
        x = get_uniform_distributed_ranges(convert_hyperparameters(selected_hyperparameter, cs), samples_per_axis=len(mean_pdp.y_variances))[0]
        ax_variances.plot(x, np.sqrt(mean_pdp.y_variances))
        # Set titles
        ax_pdp.set_title(name)
        ax_variances.set_ylabel("Std")





    # fig1.savefig("Figure 1.png")
    fig.savefig("Sampler analysis.png")
    plt.show()


if __name__ == "__main__":
    plot_sampling_bias()
