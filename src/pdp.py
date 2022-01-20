from typing import Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from src.surrogate_models import SurrogateModel


class PDP:
    def __init__(self,
                 surrogate_model: SurrogateModel,
                 cs: CS.ConfigurationSpace
                 ):
        self.surrogate_model = surrogate_model
        self.cs = cs

    def calculate_ice(self,
                      selected_hp: CSH.Hyperparameter,
                      centered: bool = False,
                      n_grid_points: int = 20,
                      n_samples: int = 1000
                      ) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Retrieve hp index from cs
        idx = self.cs.get_idx_by_hyperparameter_name(selected_hp.name)
        num_features = len(self.cs.get_hyperparameters())

        # retrieve x-values from config
        x = np.asarray([config.get_array() for config in self.cs.sample_configuration(n_samples)])
        x_s = np.linspace(0, 1, n_grid_points)

        # create x values by repeating x_s along a new dimension
        x_ice = x.repeat(n_grid_points)
        x_ice = x_ice.reshape((n_samples, num_features, n_grid_points))
        x_ice = x_ice.transpose((0, 2, 1))
        x_ice[:, :, idx] = x_s

        # predictions of surrogate
        means, stds = self.surrogate_model.predict(x_ice.reshape(-1, num_features), return_std=True)
        y_ice = means.reshape((n_samples, n_grid_points))
        stds = stds.reshape((n_samples, n_grid_points))
        variances = np.square(stds)

        # center values
        if centered:
            y_start = y_ice[:, 0].repeat(n_grid_points).reshape(n_samples, n_grid_points)
            y_ice -= y_start

        return x_ice, y_ice, variances

    def calculate_pdp(self, selected_hp: CSH.Hyperparameter, centered=False, num_grid_points: int = 20,
                      n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get all ice curves
        x_ice, y_ice, variances = self.calculate_ice(selected_hp, centered=centered, n_grid_points=num_grid_points,
                                                     n_samples=n_samples)

        # average over ice curves
        y_pdp = np.mean(y_ice, axis=0)
        x_pdp = np.mean(x_ice, axis=0)  # does not correspond to the configuration of the y_pdp-value, just an average
        variances = np.mean(variances, axis=0)

        return x_pdp, y_pdp, variances
