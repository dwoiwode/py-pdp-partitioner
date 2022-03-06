"""
A song of ICE and PDP - presented by HBOBench
"""
import warnings
from pathlib import Path

import openml
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning

from src.blackbox_functions.hpo_bench import get_SVMBenchmarkMF, get_RFBenchmarkMF
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from src.utils.advanced_plots import plot_hyperparameter_array_2D, plot_hyperparameter_array_1D

warnings.filterwarnings("ignore", category=ConvergenceWarning)

plot_folder = Path(__file__).parent.parent / "plots" / "main_hpo_bench"
plot_folder.mkdir(parents=True, exist_ok=True)

FIG_RES = 4
NUM_SAMPLES = 1000
NUM_GRID_POINTS_PER_AXIS = 50


def svm(task_id=2079):
    seed = 0
    cs, f = get_SVMBenchmarkMF(task_id, seed=seed)
    bo_sampling_points = 80
    initial_points = 12

    sampler = BayesianOptimizationSampler(f, cs, seed=seed, acq_class_kwargs={'tau': 1})
    sampler.sample(bo_sampling_points + initial_points)

    surrogate_model = GaussianProcessSurrogate(cs, seed=seed)
    surrogate_model.fit(sampler.X, sampler.y)

    # plot sampler and surrogate
    fig = plt.figure(figsize=(FIG_RES, FIG_RES))
    surrogate_model.plot_means()
    sampler.plot()
    fig.suptitle(f"SVM {task_id}: Sampler Mean")
    plt.savefig(plot_folder / f"{task_id}_svm_mean")
    plt.show()

    fig = plt.figure(figsize=(FIG_RES, FIG_RES))
    surrogate_model.plot_confidences()
    sampler.plot()
    fig.suptitle(f"SVM {task_id}: Sampler Confidence")
    plt.savefig(plot_folder / f"{task_id}_svm_confidence")
    plt.show()

    fig = plot_hyperparameter_array_1D(
        surrogate_model,
        num_samples=NUM_SAMPLES,
        num_grid_points_per_axis=NUM_GRID_POINTS_PER_AXIS,
        fig_res=FIG_RES,
        seed=seed
    )

    fig.suptitle(f"SVM {task_id}: Hyperparameters")
    fig.tight_layout()
    plt.savefig(plot_folder / f"{task_id}_svm_hyperparameters_1D")
    plt.show()

    # Hyperparameterarray
    fig_mean, fig_std = plot_hyperparameter_array_2D(
        surrogate_model,
        num_samples=NUM_SAMPLES,
        num_grid_points_per_axis=NUM_GRID_POINTS_PER_AXIS,
        fig_res=FIG_RES,
        seed=seed
    )
    fig_std.tight_layout()
    fig_mean.tight_layout()
    fig_mean.suptitle(f"SVM {task_id}: Hyperparameters (Means)")
    fig_std.suptitle(f"SVM {task_id}: Hyperparameters (Stds)")
    fig_mean.savefig(plot_folder / f"{task_id}_svm_hyperparameters_2D_mean")
    fig_std.savefig(plot_folder / f"{task_id}_svm_hyperparameters_2D_std")
    plt.show()


def rf(task_id: int = 2079):
    seed = 0
    cs, f = get_RFBenchmarkMF(task_id, seed=seed)
    bo_sampling_points = 80
    initial_points = 12

    sampler = BayesianOptimizationSampler(f, cs, seed=seed, acq_class_kwargs={'tau': 1})
    sampler.sample(bo_sampling_points + initial_points, show_progress=True)

    surrogate_model = GaussianProcessSurrogate(cs, seed=seed)
    surrogate_model.fit(sampler.X, sampler.y)

    # Hyperparameters 1D
    fig = plot_hyperparameter_array_1D(
        surrogate_model,
        num_samples=NUM_SAMPLES,
        num_grid_points_per_axis=NUM_GRID_POINTS_PER_AXIS,
        fig_res=FIG_RES,
        seed=seed
    )
    fig.suptitle(f"RF {task_id}: Hyperparameters")
    fig.tight_layout()
    plt.savefig(plot_folder / f"{task_id}_rf_hyperparameters_1D")
    plt.show()

    # Hyperparameters 2D
    fig_mean, fig_std = plot_hyperparameter_array_2D(
        surrogate_model,
        num_samples=NUM_SAMPLES,
        num_grid_points_per_axis=NUM_GRID_POINTS_PER_AXIS,
        fig_res=FIG_RES,
        seed=seed
    )
    fig_std.tight_layout()
    fig_mean.tight_layout()
    fig_mean.suptitle(f"RF {task_id}: Hyperparameters (Means)")
    fig_std.suptitle(f"RF {task_id}: Hyperparameters (Stds)")
    fig_mean.savefig(plot_folder / f"{task_id}_rf_hyperparameters_2D_mean")
    fig_std.savefig(plot_folder / f"{task_id}_rf_hyperparameters_2D_std")
    plt.show()


def analyze_openml_tasks():
    tasks = openml.tasks.list_tasks()
    numeric_only_tasks = list(
        filter(lambda t: tasks[t].get("NumberOfNumericFeatures", 0) == tasks[t].get("NumberOfFeatures", -1),
               tasks)
    )
    for i in range(20):
        print(f"{i:02d} features:")
        print(list(filter(lambda t: tasks[t]["NumberOfFeatures"] == i, numeric_only_tasks)))


if __name__ == '__main__':
    # analyze_openml_tasks()
    svm(2079)
    rf(2079)
