import logging

import matplotlib.pyplot as plt

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from src.algorithms.partitioner.random_forest_partitioner import RandomForestPartitioner  # noqa
from src.algorithms.pdp import PDP
from src.blackbox_functions import synthetic_functions
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.sampler.random_sampler import RandomSampler  # noqa
from src.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)

seed = 0
f = synthetic_functions.StyblinskiTang.for_n_dimensions(3, seed=seed)
cs = f.config_space

selected_hyperparameter = cs.get_hyperparameter("x1")

# Sampler
sampler = BayesianOptimizationSampler(f, cs, initial_points=f.ndim * 4, seed=seed)
sampler.sample(32 * f.ndim - 20, show_progress=True)  # Approximating num_samples from paper
sampler.plot(x_hyperparameters=selected_hyperparameter)

# Surrogate model
surrogate_model = GaussianProcessSurrogate(cs)
surrogate_model.fit(sampler.X, sampler.y)

# ICE
ice = ICE.from_random_points(surrogate_model, selected_hyperparameter)
ice.plot(color="orange")

# PDP
pdp = PDP.from_ICE(ice)
pdp.plot_values("black")
pdp.plot_confidences("grey")

# Partitioner
partitioner = DecisionTreePartitioner.from_ICE(ice)
# partitioner = RandomForestPartitioner.from_ICE(ice)
partitioner.partition(max_depth=2)

# Finish plot
plt.legend()
plt.tight_layout()
plt.show()
