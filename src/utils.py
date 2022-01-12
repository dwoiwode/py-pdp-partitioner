import ConfigSpace as CS
import numpy as np
from typing import List


def config_to_array(config: CS.Configuration) -> np.ndarray:
    return config.get_array()


def config_list_to_2d_arr(config_list: List[CS.Configuration]) -> np.ndarray:
    return np.asarray([config.get_array() for config in config_list])

def unscale(x: np.ndarray, cs: CS.ConfigurationSpace):
    """
    assumes that x only contains numeric values and the cs-features are located in the last dimension
    """
    x_copy = x.copy()
    num_dims = len(x.shape)
    for i, hp in enumerate(cs.get_hyperparameters()):
        if num_dims == 1:
            x_copy[i] = x[i] * (hp.upper - hp.lower) + hp.lower
        elif num_dims == 2:
            x_copy[:, i] = x[:, i] * (hp.upper - hp.lower) + hp.lower
        elif num_dims == 3:
            x_copy[:, :, i] = x[:, :, i] * (hp.upper - hp.lower) + hp.lower
        else:
            raise Exception(f'Only up to three dimensions are supported in undo_normalize, but got {num_dims}')
    return x_copy
