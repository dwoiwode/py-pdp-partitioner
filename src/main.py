import matplotlib.pyplot as plt

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.algorithms.pdp import PDP
from src.demo_data import blackbox_functions
from src.demo_data.config_spaces import config_space_nd
from src.sampler.random_sampler import RandomSampler
from src.surrogate_models import GaussianProcessSurrogate

seed = 0
f = blackbox_functions.styblinski_tang_3D
cs = config_space_nd(3, seed=seed)

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


