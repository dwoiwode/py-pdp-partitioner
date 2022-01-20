import ConfigSpace as CS
import numpy as np
from typing import List, Iterable, Union, Optional, Tuple

from ConfigSpace import hyperparameters as CSH


def config_to_array(config: CS.Configuration) -> np.ndarray:
    return config.get_array()


def convert_config_list_to_np(X: Union[List[CS.Configuration], List[Iterable[float]], np.ndarray]) -> np.ndarray:
    if isinstance(X, list):
        if isinstance(X[0], CS.Configuration):
            X = [x.values() for x in X]
    return np.asarray(X)


def config_list_to_2d_arr(config_list: List[CS.Configuration]) -> np.ndarray:
    return np.asarray([config.get_array() for config in config_list])


def get_selected_idx(selected_hyperparameter: Iterable[CSH.Hyperparameter],
                     config_space: CS.ConfigurationSpace) -> List[int]:
    return [
        config_space.get_idx_by_hyperparameter_name(hp.name)
        for hp in selected_hyperparameter
    ]


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


def get_stds(stds: Optional[np.ndarray] = None, variances: Optional[np.ndarray] = None) -> np.ndarray:
    if stds is None and variances is None or (stds is not None and variances is not None):
        raise RuntimeError("Requires either stds or variances")

    if variances is not None:
        stds = np.sqrt(variances)
    return stds


def get_hyperparameters(hyperparameters: Union[None, CSH.Hyperparameter, Iterable[CSH.Hyperparameter]],
                        cs: CS.ConfigurationSpace) -> List[CSH.Hyperparameter]:
    if hyperparameters is None:
        return list(cs.get_hyperparameters())
    elif isinstance(hyperparameters, CSH.Hyperparameter):
        return [hyperparameters]
    else:
        return list(hyperparameters)


def get_uniform_distributed_ranges(cs: CS.ConfigurationSpace,
                                   samples_per_axis: int = 100,
                                   scaled=False) -> np.ndarray:
    """
    :param cs: Configuration_space to sample from
    :param samples_per_axis: Number of samples per axis
    :param scaled: if scaled: Ranges normalized between 0 and 1, otherwise Ranges are as give n in configspace
    :return: Shape: (num_hyperparameters, num_samples_per_axis)
    """
    ranges = []
    for parameter in cs.get_hyperparameters():
        assert isinstance(parameter, CSH.NumericalHyperparameter)
        if scaled:
            ranges.append(np.linspace(0, 1, num=samples_per_axis))
        else:
            if parameter.log:
                space_function = np.logspace
            else:
                space_function = np.linspace

            ranges.append(space_function(parameter.lower, parameter.upper, num=samples_per_axis))
    res = np.asarray(ranges)
    assert len(res) == len(cs.get_hyperparameters())
    return res
