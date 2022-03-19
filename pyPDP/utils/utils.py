import hashlib
import logging
import math
import time
from abc import ABC
from typing import List, Iterable, Union, Optional

import ConfigSpace as CS
import numpy as np
from ConfigSpace import hyperparameters as CSH

from pyPDP.utils.typing import SelectedHyperparameterType


class ConfigSpaceHolder(ABC):
    def __init__(self, config_space: CS.ConfigurationSpace, *, seed: Union[None, int, bool] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"{self.__class__.__name__}: Initial Seed = {seed}")

        if seed is True:
            # Use existing config_space
            self.logger.debug(f"{self.__class__.__name__}: Seed = Use existing")
            self.config_space = config_space
            return
        elif seed is None:
            # Use random seed
            seed = int(time.time() * 1000)
        else:
            # Use seed
            # Hash of class prevents using the exact same seed for every step (e.g. Sampler, ICE, ...)
            h = hashlib.md5(self.__class__.__name__.encode("latin"), usedforsecurity=False)
            seed = int.from_bytes(h.digest(), "big") + seed
        self.logger.debug(f"{self.__class__.__name__}: Final Seed = {seed % 2 ** 31}")
        self.config_space = copy_config_space(config_space, seed=seed % 2 ** 31)

    def sample_random_configuration(self, n: int) -> List[CS.Configuration]:
        return self.config_space.sample_configuration(n)


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


def scale_float(
        value: float,
        cs: CS.ConfigurationSpace,
        hp: CSH.NumericalHyperparameter
) -> float:
    cs_hp = cs.get_hyperparameter(hp.name)
    if cs_hp.log:
        log_lower = np.log(cs_hp.lower)
        log_upper = np.log(cs_hp.upper)
        value = np.log(value)
        normalized_value = (value - log_lower) / (log_upper - log_lower)
    else:
        normalized_value = (value - cs_hp.lower) / (cs_hp.upper - cs_hp.lower)
    normalized_value = np.minimum(1, normalized_value)
    normalized_value = np.maximum(0, normalized_value)
    return normalized_value


def unscale_float(
        normalized_value: float,
        cs: CS.ConfigurationSpace,
        hp: CSH.Hyperparameter
) -> float:
    cs_hp = cs.get_hyperparameter(hp.name)
    if cs_hp.log:
        log_lower = np.log(cs_hp.lower)
        log_upper = np.log(cs_hp.upper)
        value = normalized_value * (log_upper - log_lower) + log_lower
        value = math.exp(value)
    else:
        value = normalized_value * (cs_hp.upper - cs_hp.lower) + cs_hp.lower
    value = np.minimum(cs_hp.upper, np.maximum(cs_hp.lower, value))
    return value


def unscale(x: np.ndarray, cs: CS.ConfigurationSpace) -> np.ndarray:
    """
    assumes that the cs-features are located in the last dimension
    """
    x_copy = x.copy()
    num_dims = len(x.shape)
    for i, hp in enumerate(cs.get_hyperparameters()):
        if isinstance(hp, CSH.NumericalHyperparameter):
            if hp.log:
                unscaler = lambda values: \
                    np.minimum(hp.upper,
                               np.maximum(hp.lower,
                                          np.exp(values * (np.log(hp.upper) - np.log(hp.lower)) + np.log(hp.lower))))
            else:
                unscaler = lambda values: values * (hp.upper - hp.lower) + hp.lower
        elif isinstance(hp, CSH.CategoricalHyperparameter):
            unscaler = lambda values: np.asarray([hp.choices[k] for k in values])
            x_copy = x_copy.astype(object)
        else:
            raise TypeError(f"Currently not support hyperparameter-type {type(hp)}")

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


def get_hyperparameters(hyperparameters: Optional[SelectedHyperparameterType],
                        cs: CS.ConfigurationSpace) -> List[CSH.Hyperparameter]:
    if hyperparameters is None:
        # None -> All hyperparameters in cs
        return list(cs.get_hyperparameters())
    elif isinstance(hyperparameters, CSH.Hyperparameter):
        # Single Hyperparameter
        return [hyperparameters]
    elif isinstance(hyperparameters, str):
        # Single Hyperparameter name
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


def get_uniform_distributed_ranges(
        cs: Union[CS.ConfigurationSpace, Iterable[CSH.NumericalHyperparameter]],
        samples_per_axis: int = 100,
        scaled=False
) -> np.ndarray:
    """
    :param cs: Configuration_space to sample from
    :param samples_per_axis: Number of samples per axis
    :param scaled: if scaled: Ranges normalized between 0 and 1, otherwise Ranges are as give n in configspace
    :return: Shape: (num_hyperparameters, num_samples_per_axis)
    """
    ranges = []
    if isinstance(cs, CS.ConfigurationSpace):
        cs = cs.get_hyperparameters()
    for parameter in cs:
        assert isinstance(parameter, CSH.NumericalHyperparameter)
        if scaled:
            ranges.append(np.linspace(0, 1, num=samples_per_axis))
        else:
            if parameter.log:
                ranges.append(np.logspace(np.log10(parameter.lower), np.log10(parameter.upper), num=samples_per_axis))
            else:
                ranges.append(np.linspace(parameter.lower, parameter.upper, num=samples_per_axis))

    res = np.asarray(ranges)
    assert len(res) == len(cs)
    return res


def median_distance_between_points(X: np.ndarray) -> float:
    X = np.expand_dims(X, axis=0)
    dif = X - X.transpose((1, 0, 2))
    dif_2 = np.sum(np.square(dif), axis=2)
    distances = np.sqrt(dif_2)
    median = np.median(distances[distances != 0]).item()
    return median


def calculate_log_delta(nll: float, nll_root: float) -> float:
    return (nll_root - nll) / np.absolute(nll_root)


def convert_hyperparameters(
        hyperparameters: Union[str, CSH.Hyperparameter, Iterable[Union[CSH.Hyperparameter, str]]],
        config_space: CS.ConfigurationSpace
) -> List[CSH.Hyperparameter]:
    """
    Converts either
        * a single hyperparameter (CSH.Hyperparameter)
        * a single hyperparameter name (str)
        * an iterable (list, tuple, etc.) of hyperparameters (CSH.Hyperparameter)
        * an iterable of hyperparameter names (str)
    to a list of hyperparameters (CSH.Hyperparameter)
    """
    if isinstance(hyperparameters, (str, CSH.Hyperparameter)):
        hyperparameters = (hyperparameters,)
    hps = []
    for hp in hyperparameters:
        if isinstance(hp, str):
            hps.append(config_space.get_hyperparameter(hp))
        elif isinstance(hp, CSH.Hyperparameter):
            hps.append(hp)
        else:
            raise TypeError(f'Could not interpret hp: {hp}')
    return hps


def copy_config_space(cs: CS.ConfigurationSpace, *, seed=None) -> CS.ConfigurationSpace:
    # copy cs
    hp_dic = {}
    for hp in cs.get_hyperparameters():
        if isinstance(hp, CSH.NumericalHyperparameter):
            new_hp = hp.__class__(hp.name, lower=hp.lower, upper=hp.upper, log=hp.log)
            hp_dic[hp.name] = new_hp
        elif isinstance(hp, CSH.CategoricalHyperparameter):
            new_hp = hp.__class__(hp.name, choices=hp.choices[:])  # Copy choices
            hp_dic[hp.name] = new_hp  # TODO: Test copy categorical hp and unscaler
        else:
            raise TypeError(f"Currently not support hyperparameter-type {type(hp)}")

    # add new hp to new cs
    cs_copy = CS.ConfigurationSpace(seed=seed)
    for hp in hp_dic.values():
        cs_copy.add_hyperparameter(hp)

    return cs_copy


class ProgressDummy:
    """ Small dummy class that does nothing except existing """

    def update(self, n=1):
        pass

    def close(self):
        pass

    def refresh(self, nolock=False, lock_args=None):
        pass
