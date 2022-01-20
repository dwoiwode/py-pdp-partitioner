from dataclasses import dataclass
from typing import Union, Iterable, Optional

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from src.algorithms import Algorithm, Plottable
from src.surrogate_models import SurrogateModel


@dataclass
class ICECurve(Plottable):
    def __init__(self, x_ice: np.ndarray, y_ice: np.ndarray, y_variances: np.ndarray):
        """
        :param x_ice: Shape: (num_gridpoints, num_features)
        :param y_ice: Shape: (num_gridpoints)
        :param y_variances: (num_gridpoints)
        """
        self.x_ice: np.ndarray
        self.y_ice: np.ndarray
        self.y_variances: np.ndarray

    def plot(self, color="red", with_confidence=False, ax=None):
        pass

    @property
    def config(self) -> CS.Configuration:
        # TODO: Config (alle fix außer selected hyperparameter)
        return


class ICE(Algorithm):
    def __init__(self, surrogate_model: SurrogateModel,
                 selected_hyperparameter: Union[CSH.Hyperparameter, Iterable[CSH.Hyperparameter]],
                 num_grid_points_per_axis: int = 20,
                 num_samples: int = 1000):
        super().__init__(surrogate_model, selected_hyperparameter)
        self.n_selected_hyperparameter = len(self.selected_hyperparameter)
        self.num_grid_points_per_axis = num_grid_points_per_axis
        self.num_grid_points = num_grid_points_per_axis * self.n_selected_hyperparemeter
        self.num_samples = num_samples

        # Properties
        self._x_ice: Optional[np.ndarray] = None
        self._y_ice: Optional[np.ndarray] = None
        self._y_variances: Optional[np.ndarray] = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return ICECurve(self.x_ice[idx], self.y_ice[idx], self.y_variances[idx])

    def _calculate(self):
        pass
        # # Retrieve hp index from cs
        # idx = self.cs.get_idx_by_hyperparameter_name(selected_hp.name)
        # num_features = len(self.cs.get_hyperparameters())
        #
        # # retrieve x-values from config
        # x = np.asarray([config.get_array() for config in self.cs.sample_configuration(n_samples)])
        # x_s = np.linspace(0, 1, n_grid_points)
        #
        # # create x values by repeating x_s along a new dimension
        # x_ice = x.repeat(n_grid_points)
        # x_ice = x_ice.reshape((n_samples, num_features, n_grid_points))
        # x_ice = x_ice.transpose((0, 2, 1))
        # x_ice[:, :, idx] = x_s
        #
        # # predictions of surrogate
        # means, stds = self.surrogate_model.predict(x_ice.reshape(-1, num_features), return_std=True)
        # y_ice = means.reshape((n_samples, n_grid_points))
        # stds = stds.reshape((n_samples, n_grid_points))
        # variances = np.square(stds)
        #
        # # center values
        # if centered:
        #     y_start = y_ice[:, 0].repeat(n_grid_points).reshape(n_samples, n_grid_points)
        #     y_ice -= y_start
        #
        # return x_ice, y_ice, variances

    @property
    def x_ice(self) -> np.ndarray:
        if self._x_ice is None:
            self._calculate()
        return self._x_ice

    @property
    def y_ice(self) -> np.ndarray:
        if self._y_ice is None:
            self._calculate()
        return self._y_ice

    @property
    def y_variances(self) -> np.ndarray:
        if self._y_variances is None:
            self._calculate()
        return self._y_variances

    def plot(self, color="red", ax=None):
        pass
