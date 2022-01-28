from typing import Callable

import ConfigSpace as CS
import numpy as np

from src.sampler import Sampler


class RandomSampler(Sampler):
    def __init__(self,
                 obj_func: Callable,
                 config_space: CS.ConfigurationSpace,
                 minimize_objective=True,
                 seed=None):
        super().__init__(obj_func, config_space, minimize_objective=minimize_objective, seed=seed)
        self.rng = np.random.RandomState(self.seed)  # Create new rng, so we have control over seed

    def sample(self, n_points: int = 1):
        self.logger.debug(f"Sample {n_points}")
        n_features = len(self.config_space.get_hyperparameters())
        samples = self.rng.random((n_points, n_features))
        origin = self.__class__.__name__
        for i in range(n_points):
            config = CS.Configuration(self.config_space, vector=samples[i], origin=origin)
            # config = self.config_space.sample_configuration()
            value = self.obj_func(**config)
            self.config_list.append(config)
            self.y_list.append(value)
