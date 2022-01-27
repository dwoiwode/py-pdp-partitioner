from abc import ABC

import ConfigSpace.hyperparameters as CSH

from src.surrogate_models import SurrogateModel
from src.utils.plotting import Plottable
from src.utils.typing import SelectedHyperparameterType
from src.utils.utils import convert_hyperparameters


class Algorithm(Plottable, ABC):
    def __init__(self,
                 surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 num_samples: int = 1000,
                 num_grid_points_per_axis: int = 20,
                 seed=None,
                 ):
        super().__init__()
        self.surrogate_model = surrogate_model
        self.config_space = surrogate_model.config_space
        self.num_samples = num_samples
        self.num_grid_points_per_axis = num_grid_points_per_axis
        self.seed = seed

        self.selected_hyperparameter = convert_hyperparameters(selected_hyperparameter, self.config_space)

        self.num_grid_points = num_grid_points_per_axis * self.n_selected_hyperparameter

    @property
    def n_selected_hyperparameter(self) -> int:
        return len(self.selected_hyperparameter)

    @property
    def num_features(self) -> int:
        return len(self.config_space.get_hyperparameters())
