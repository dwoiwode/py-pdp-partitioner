from src.demo_data import blackbox_functions
from src.demo_data.config_spaces import config_space_nd
from src.sampler import RandomSampler
from src.surrogate_models import GaussianProcessSurrogate

f = blackbox_functions.styblinski_tang_3D
cs = config_space_nd(3)

sampler = RandomSampler(f, cs)
sampler.sample(150)

surrogate_model = GaussianProcessSurrogate(cs)
surrogate_model.fit(sampler.X, sampler.y)

selected_hyperparamter = cs.get_hyperparameter("x1")

# ice = ICE(surrogate_model, selected_hyperparamter)
