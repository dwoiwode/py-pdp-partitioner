from typing import Optional, Callable, Any, Type, Tuple
import ConfigSpace as CS

from src.optimizer import AbstractOptimizer, RandomSearch
from src.partitioner import AbstractPartitioner, DecisionTreePartitioner

import numpy as np


class PDP:
    def __init__(self,
                 # partitioner: Optional[AbstractPartitioner],
                 optimizer: Optional[AbstractOptimizer]):
        # self.partitioner = partitioner
        self.optimizer = optimizer

    def calculate_ice(self, idx: int, centered: bool = False, num_grid_points: int = 1000) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # retrieve x-values from config
        x = np.asarray([config.get_array() for config in self.optimizer.config_list])
        num_instances, num_features = x.shape
        x_s = np.linspace(0, 1, num_grid_points)

        # create x values by repeating x_s along a new dimension
        x_ice = x.repeat(num_grid_points)
        # x_ice = x_ice.reshape((num_instances, num_grid_points, num_features))
        x_ice = x_ice.reshape((num_instances, num_features, num_grid_points))
        x_ice = x_ice.transpose((0, 2, 1))
        x_ice[:, :, idx] = x_s

        # predictions of surrogate
        means, stds = self.optimizer.surrogate_score(x_ice.reshape(-1, num_features))
        y_ice = means.reshape((num_instances, num_grid_points))
        stds = stds.reshape((num_instances, num_grid_points))
        variances = np.square(stds)

        # center values
        if centered:
            y_start = y_ice[:, 0].repeat(num_grid_points).reshape(num_instances, num_grid_points)
            y_ice -= y_start

        return x_ice, y_ice, variances

    def calculate_pdp(self, idx: int, centered=False, num_grid_points: int = 1000) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get all ice curves
        x_ice, y_ice, variances = self.calculate_ice(idx, centered=centered, num_grid_points=num_grid_points)

        # average over ice curves
        y_pdp = np.mean(y_ice, axis=0)
        x_pdp = np.mean(x_ice, axis=0)  # does not correspond to the configuration of the y_pdp-value, just an average
        variances = np.mean(variances, axis=0)

        return x_pdp, y_pdp, variances
