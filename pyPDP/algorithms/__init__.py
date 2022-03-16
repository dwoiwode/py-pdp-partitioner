from abc import ABC

import numpy as np

from pyPDP.surrogate_models import SurrogateModel
from pyPDP.utils.typing import SelectedHyperparameterType
from pyPDP.utils.utils import convert_hyperparameters, ConfigSpaceHolder


class Algorithm(ConfigSpaceHolder, ABC):
    def __init__(self,
                 surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 samples: np.ndarray,
                 num_grid_points_per_axis: int = 20,
                 seed=None,
                 ):
        super().__init__(surrogate_model.config_space, seed=seed)
        self.surrogate_model = surrogate_model
        self.samples = samples
        self.num_samples = len(samples)
        self.num_grid_points_per_axis = num_grid_points_per_axis

        self.selected_hyperparameter = convert_hyperparameters(selected_hyperparameter, self.config_space)

        self.num_grid_points = num_grid_points_per_axis ** self.n_selected_hyperparameter

    @classmethod
    def from_random_points(cls,
                           surrogate_model: SurrogateModel,
                           selected_hyperparameter: SelectedHyperparameterType,
                           num_samples: int = 1000,
                           num_grid_points_per_axis: int = 20,
                           seed=None):
        samples = np.asarray([
            config.get_array()
            for config in surrogate_model.config_space.sample_configuration(num_samples)
        ])
        return cls(
            surrogate_model=surrogate_model,
            selected_hyperparameter=selected_hyperparameter,
            samples=samples,
            num_grid_points_per_axis=num_grid_points_per_axis,
            seed=seed
        )

    @property
    def n_selected_hyperparameter(self) -> int:
        return len(self.selected_hyperparameter)

    @property
    def num_features(self) -> int:
        return len(self.config_space.get_hyperparameters())
