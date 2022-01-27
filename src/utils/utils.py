from typing import List, Iterable, Union, Optional

import ConfigSpace as CS
import numpy as np
from ConfigSpace import hyperparameters as CSH


def config_to_array(config: CS.Configuration) -> np.ndarray:
    return config.get_array()


def config_list_to_array(X: Union[List[CS.Configuration], List[Iterable[float]], np.ndarray]) -> np.ndarray:
    if isinstance(X, list):
        if isinstance(X[0], CS.Configuration):
            X = [config.get_array() for config in X]
    return np.asarray(X)


def get_selected_idx(selected_hyperparameter: Iterable[CSH.Hyperparameter],
                     config_space: CS.ConfigurationSpace) -> List[int]:
    return [
        config_space.get_idx_by_hyperparameter_name(hp.name)
        for hp in selected_hyperparameter
    ]


def scale_float(value: float, cs: CS.ConfigurationSpace, hp: CSH.Hyperparameter):
    cs_hp = cs.get_hyperparameter(hp.name)
    normalized_value = (value - cs_hp.lower) / (cs_hp.upper - cs_hp.lower)
    if hp.log:
        return np.log10(normalized_value)
    return normalized_value


def unscale_float(normalized_value: float, cs: CS.ConfigurationSpace, hp: CSH.Hyperparameter):
    cs_hp = cs.get_hyperparameter(hp.name)
    value = normalized_value * (cs_hp.upper - cs_hp.lower) + cs_hp.lower
    return value


def unscale(x: np.ndarray, cs: CS.ConfigurationSpace):
    """
    assumes that x only contains numeric values and the cs-features are located in the last dimension
    """
    x_copy = x.copy()
    num_dims = len(x.shape)
    for i, hp in enumerate(cs.get_hyperparameters()):
        assert isinstance(hp, CSH.NumericalHyperparameter), "Currently only Numerical Hyperparameters are supported"
        if hp.log:
            unscaler = lambda values: np.power(10, values * (hp.upper - hp.lower) + hp.lower)
        else:
            unscaler = lambda values: values * (hp.upper - hp.lower) + hp.lower
        if num_dims == 1:
            x_copy[i] = unscaler(x[i])
        elif num_dims == 2:
            x_copy[:, i] = unscaler(x[:, i])
        elif num_dims == 3:
            x_copy[:, :, i] = unscaler(x[:, :, i])
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
    elif isinstance(hyperparameters, str):
        return [cs.get_hyperparameter(hyperparameters)]
    else:
        # Either list of names or list of Hyperparameters
        hps = []
        for hp in hyperparameters:
            if isinstance(hp, str):
                hps.append(cs.get_hyperparameter(hp))
            elif isinstance(hp, CSH.Hyperparameter):
                hps.append(hp)
            else:
                raise TypeError(f"Could not identify hyperparameter {hp} (Type: {type(hp)})")
        return hps


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
                ranges.append(np.logspace(np.log10(parameter.lower), np.log10(parameter.upper), num=samples_per_axis))
            else:
                ranges.append(np.linspace(parameter.lower, parameter.upper, num=samples_per_axis))

    res = np.asarray(ranges)
    assert len(res) == len(cs.get_hyperparameters())
    return res


def median_distance_between_points(X: np.ndarray) -> float:
    X = np.expand_dims(X, axis=0)
    dif = X - X.transpose((1, 0, 2))
    dif_2 = np.sum(np.square(dif), axis=2)
    distances = np.sqrt(dif_2)
    median = np.median(distances[distances != 0]).item()
    return median

def convert_hyperparameters(hyperparameters: Iterable[Union[CSH.Hyperparameter, str]],
                            config_space: CS.ConfigurationSpace) -> List[CSH.Hyperparameter]:
    hps = []
    for hp in hyperparameters:
        if isinstance(hp, str):
            hps.append(config_space.get_hyperparameter(hp))
        elif isinstance(hp, CSH.Hyperparameter):
            hps.append(hp)
        else:
            raise TypeError(f'Could not interpret hp: {hp}')
    return hps
