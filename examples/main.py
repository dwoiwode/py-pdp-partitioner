import logging

import matplotlib.pyplot as plt

from pyPDP.algorithms.ice import ICE
from pyPDP.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from pyPDP.algorithms.partitioner.random_forest_partitioner import RandomForestPartitioner  # noqa
from pyPDP.algorithms.pdp import PDP
from pyPDP.blackbox_functions import synthetic_functions
from pyPDP.sampler.bayesian_optimization import BayesianOptimizationSampler
from pyPDP.sampler.random_sampler import RandomSampler  # noqa
from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from utils.utils import calculate_log_delta

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)

# Define Blackbox function
seed = 0
f = synthetic_functions.StyblinskiTang.for_n_dimensions(3, seed=seed)
cs = f.config_space

selected_hyperparameter = cs.get_hyperparameter("x1")

# Sampler
sampler = BayesianOptimizationSampler(f, cs, initial_points=f.ndim * 4, seed=seed)
# Approximating num_samples from paper
#  d  |  n_samples
# --- | -----------
#  d  |   32 * d - 20
#  1  |   12
#  3  |   86
#  5  |   140
#  7  |   202
num_samples = 32 * f.ndim - 20
sampler.sample(num_samples, show_progress=True)
sampler.plot(x_hyperparameters=selected_hyperparameter)

# Surrogate model
surrogate_model = GaussianProcessSurrogate(cs)
surrogate_model.fit(sampler.X, sampler.y)

# ICE
ice = ICE.from_random_points(surrogate_model, selected_hyperparameter)
# ice.plot(color="orange")

# PDP
pdp = PDP.from_ICE(ice)
pdp.plot_values("black")
pdp.plot_confidences("grey")

# Partitioner
partitioner = DecisionTreePartitioner.from_ICE(ice)
# partitioner = RandomForestPartitioner.from_ICE(ice)
partitioner.partition(max_depth=2)
incumbent_region = partitioner.get_incumbent_region(sampler.incumbent_config)
print(incumbent_region.implied_config_space())

# true_pd = f.pd_integral(selected_hyperparameter, seed=seed)
nll_base = partitioner.root.negative_log_likelihood(f)
nll_leaf = incumbent_region.negative_log_likelihood(f)
mc_base = partitioner.root.mean_confidence
mc_leaf = incumbent_region.mean_confidence

print("NLL Base:", nll_base)
print("NLL Incumbent:", nll_leaf)
print("MC Base:", mc_base)
print("MC Incumbent:", mc_leaf)

nll_log_delta = calculate_log_delta(nll_leaf, nll_base)
mc_log_delta = calculate_log_delta(mc_leaf, mc_base)
print("NLL Improvement:", nll_log_delta)
print("MC Improvement:", mc_log_delta)

# Finish plot
plt.legend()
plt.tight_layout()
plt.show()
