import matplotlib.pyplot as plt

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.algorithms.pdp import PDP
from src.blackbox_functions import synthetic_functions
from src.sampler.random_sampler import RandomSampler
from src.surrogate_models import GaussianProcessSurrogate

seed = 0  # TODO: Use seed!!
f = synthetic_functions.StyblinskiTang.for_n_dimensions(3)
cs = f.config_space

selected_hyperparameter = cs.get_hyperparameter("x1")

# Sampler
sampler = RandomSampler(f, cs, seed=seed)
sampler.sample(100)
sampler.plot(x_hyperparameters=selected_hyperparameter)

# Surrogate model
surrogate_model = GaussianProcessSurrogate(cs)
surrogate_model.fit(sampler.X, sampler.y)

# ICE
ice = ICE(surrogate_model, selected_hyperparameter)
ice.plot(color="orange")

# PDP
pdp = PDP.from_ICE(ice)
pdp.plot("black", "grey", with_confidence=True)

# Partitioner
dt_partitioner = DTPartitioner(surrogate_model, selected_hyperparameter)
dt_partitioner.partition(max_depth=2)
# dt_partitioner.plot()

# Finish plot
plt.legend()
plt.tight_layout()
plt.show()


