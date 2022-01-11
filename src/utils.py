import ConfigSpace as CS
import numpy as np
from typing import List


def config_to_array(config: CS.Configuration) -> np.ndarray:
    # return np.asarray(list(config.get_dictionary().values()))
    return config.get_array()


def config_list_to_2d_arr(config_list: List[CS.Configuration]) -> np.ndarray:
    return np.asarray([config_to_array(config) for config in config_list])
