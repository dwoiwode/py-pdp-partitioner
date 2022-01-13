from typing import Optional, Callable, Any, Type, Tuple
import ConfigSpace as CS

from src.optimizer import AbstractOptimizer, RandomSearch
from src.partitioner import AbstractPartitioner, DecisionTreePartitioner

import numpy as np


class PDP:
    def __init__(self,
                 partitioner: Optional[AbstractPartitioner],
                 optimizer: Optional[AbstractOptimizer]):
        self.partitioner = partitioner
        self.optimizer = optimizer

    def calculate_ice(self, idx: int, ordered=True, centered=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # retrieve x-values from config
        x = np.asarray([config.get_array() for config in self.optimizer.config_list])
        num_instances, num_features = x.shape
        x_s = x[:, idx]

        # order values according to x_s (only works with single x_s)
        if ordered:
            order = np.argsort(x_s)
            x_s = x_s[order]
            x = x[order]

        # create x values by repeating x_s along a new dimension
        x_ice = x.repeat(num_instances).reshape((num_instances, num_features, -1)).transpose((2, 0, 1))
        for i in range(num_instances):
            x_ice[i, :, idx] = x_s[i]
        means, stds = self.optimizer.surrogate_score(x_ice.reshape(-1, num_features))
        y_ice = means.reshape((num_instances, num_instances))
        stds = stds.reshape((num_instances, num_instances))
        variances = np.square(stds)

        # center values
        if centered:
            y_start = y_ice[0, :].repeat(num_instances).reshape(num_instances, num_instances).T
            y_ice -= y_start

        # make every ice curve accessible in first dimension
        x_ice = x_ice.transpose((1, 0, 2))
        y_ice = y_ice.T

        return x_ice, y_ice, variances

    def calculate_pdp(self, idx: int, centered=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get all ice curves
        x_ice, y_ice, variances = self.calculate_ice(idx, ordered=True, centered=centered)

        # average over ice curves
        y_pdp = np.mean(y_ice, axis=0)
        x_pdp = np.mean(x_ice, axis=0)  # does not correspond to the configuration of the y_pdp-value, just an average
        variances = np.mean(variances, axis=0)

        return x_pdp, y_pdp, variances
