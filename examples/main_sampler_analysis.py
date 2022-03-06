import warnings
from pathlib import Path
from typing import Dict, Type, Any, Callable

import numpy as np
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from src.algorithms.ice import ICECurve
from src.algorithms.pdp import PDP
from src.blackbox_functions import BlackboxFunctionND
from src.blackbox_functions.synthetic_functions import StyblinskiTang
from src.sampler import Sampler
from src.sampler.acquisition_function import LowerConfidenceBound
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.sampler.grid_sampler import GridSampler
from src.sampler.random_sampler import RandomSampler
from src.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from src.utils.plotting import plot_function
from src.utils.utils import get_uniform_distributed_ranges, convert_hyperparameters

warnings.filterwarnings("ignore", category=ConvergenceWarning)
seed = 0

plot_folder = Path(__file__).parent.parent / "plots" / "sampler_analysis"
plot_folder.mkdir(parents=True, exist_ok=True)


def plot_sampling_bias(
        figure_name: str,
        f_class: Type[BlackboxFunctionND] = StyblinskiTang,
        dimensions=2,
        sampler_factories: Dict[str, Callable[[Any], Sampler]] = None,
        sampled_points=64 - 8,
        n_repetitions=1,
        seed_offset=seed,
):
    """
    Takes `sampler_factories` as input and plot their pdps averaged over `n_repetitions`.

    `sampler_factories` is a dict:
        key = name of function
        value = Function that takes seed as input and returns a Sampler
    """
    f = f_class.for_n_dimensions(dimensions)
    cs = f.config_space
    initial_points = 4 * f.ndim
    if sampler_factories is None:
        sampler_factories = {
            "High sampling bias": lambda s: BayesianOptimizationSampler(
                f,
                cs,
                initial_points=initial_points,
                acq_class=LowerConfidenceBound,
                acq_class_kwargs={"tau": 0.1},
                seed=s
            ),
            "Medium Sampling bias": lambda s: BayesianOptimizationSampler(
                f,
                cs,
                initial_points=initial_points,
                acq_class=LowerConfidenceBound,
                acq_class_kwargs={"tau": 2},
                seed=s
            ),
            "Low Sampling bias": lambda s: BayesianOptimizationSampler(
                f,
                cs,
                initial_points=initial_points,
                acq_class=LowerConfidenceBound,
                acq_class_kwargs={"tau": 5},
                seed=s
            ),
            "Random": lambda s: RandomSampler(
                f,
                cs,
                seed=s
            ),
            "Grid": lambda s: GridSampler(
                f,
                cs,
                seed=s
            ),
        }

    selected_hyperparameter = ["x1"]

    n = len(sampler_factories)
    fig, axes = plt.subplots(2, n, sharex="all", sharey="row", figsize=(2.5 * n, 5))

    for (name, sampler_factory), ax3 in zip(sampler_factories.items(), axes.T):
        arr_means = []
        arr_variances = []
        arr_x = []
        arr_mmd = []

        for i in tqdm(range(n_repetitions), desc=f"Sampler: {name}"):
            s = seed_offset + i
            f = f_class.for_n_dimensions(dimensions, seed=seed)
            cs = f.config_space

            sampler = sampler_factory(s)
            assert isinstance(sampler, Sampler)
            sampler.sample(sampled_points + initial_points)
            surrogate = GaussianProcessSurrogate(cs, seed=s)
            surrogate.fit(sampler.X, sampler.y)
            pdp = PDP.from_random_points(
                surrogate_model=surrogate, selected_hyperparameter=selected_hyperparameter,
                num_grid_points_per_axis=50, num_samples=4000,
            )
            arr_x = pdp.x_pdp
            arr_means.append(pdp.y_pdp)
            arr_variances.append(pdp.y_variances)
            arr_mmd.append(sampler.maximum_mean_discrepancy(seed=s + 1))

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
        mean_pdp.plot_values(color="blue", ax=ax_pdp)
        mean_pdp.plot_confidences(
            line_color="blue",
            gradient_color="lightblue",
            confidence_max_sigma=3,
            ax=ax_pdp
        )

        f_pd = f.pd_integral(*[hp for hp in cs if hp not in selected_hyperparameter])
        plot_function(f_pd, f_pd.config_space, samples_per_axis=200, ax=ax_pdp)

        # Plot variances
        x = get_uniform_distributed_ranges(
            convert_hyperparameters(selected_hyperparameter, cs),
            samples_per_axis=len(mean_pdp.y_variances)
        )[0]
        ax_variances.plot(x, np.sqrt(mean_pdp.y_variances))
        # Set titles
        ax_pdp.set_title(f"{name}\n(mmd={np.mean(arr_mmd):.2f}$\pm${np.std(arr_mmd):.2f})")
        ax_variances.set_ylabel("Std")

    # fig1.savefig("Figure 1.png")
    fig.savefig(plot_folder / figure_name)
    plt.show()


if __name__ == "__main__":
    # styblinski-tang 2d
    plot_sampling_bias('Styblinski-Tang-10-Rep', n_repetitions=10)
