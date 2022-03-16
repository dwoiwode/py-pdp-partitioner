import random
from typing import Callable, List, Optional, Union

import ConfigSpace as CS
import numpy as np
from ConfigSpace.util import generate_grid
from tqdm import tqdm

from pyPDP.sampler import Sampler
from pyPDP.utils.utils import ProgressDummy


class GridSampler(Sampler):
    def __init__(
            self,
            obj_func: Callable,
            config_space: CS.ConfigurationSpace,
            minimize_objective=True,
            seed=None
    ):
        super().__init__(
            obj_func=obj_func,
            config_space=config_space,
            minimize_objective=minimize_objective,
            seed=seed
        )
        self._grid: Optional[List[CS.Configuration]] = None
        self.rng = random.Random()
        self.rng.seed(seed)

    def _sample(self, n_points: int = 1, pbar: Union[ProgressDummy, tqdm] = ProgressDummy()):
        expected_length = len(self) + n_points
        if self._grid is None or len(self) + len(self._grid) < expected_length:
            n_dims = len(self.config_space.get_hyperparameters())
            samplers_per_axis = int(np.ceil(expected_length ** (1 / n_dims)))
            num_steps_dict = {param.name: samplers_per_axis for param in self.config_space.get_hyperparameters()}
            self._grid = generate_grid(self.config_space, num_steps_dict)
            self.rng.shuffle(self._grid)

        while len(self) < expected_length:
            sample = self._grid.pop(0)
            if sample in self.config_list:
                continue
            self.config_list.append(sample)
            self.y_list.append(self.obj_func(**sample))

            pbar.update(1)
            pbar.refresh()
