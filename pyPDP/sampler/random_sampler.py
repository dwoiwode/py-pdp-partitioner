from typing import Callable, Union

import ConfigSpace as CS
from tqdm import tqdm

from pyPDP.sampler import Sampler
from pyPDP.utils.utils import ProgressDummy


class RandomSampler(Sampler):
    def __init__(
            self,
            obj_func: Callable,
            config_space: CS.ConfigurationSpace,
            minimize_objective=True,
            seed=None
    ):
        super().__init__(obj_func, config_space, minimize_objective=minimize_objective, seed=seed)

    def _sample(self, n_points: int = 1, pbar: Union[ProgressDummy, tqdm] = ProgressDummy()):
        self.logger.debug(f"Random Sample {n_points}")
        samples = self.config_space.sample_configuration(n_points)
        for i in range(n_points):
            config = samples[i]
            value = self.obj_func(**config)
            self.config_list.append(config)
            self.y_list.append(value)
            pbar.update(1)
            pbar.refresh()
