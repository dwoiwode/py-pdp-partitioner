import ConfigSpace as CS
import numpy as np


def config_to_array(config: CS.Configuration) -> np.ndarray:
    return np.asarray(list(config.get_dictionary().values()))


def config_list_to_2d_arr(config_list: list[CS.Configuration]) -> np.ndarray:
    return np.asarray([config_to_array(config) for config in config_list])
