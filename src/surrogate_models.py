from abc import ABC, abstractmethod
from typing import Union, List, Tuple

import ConfigSpace as CS

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import convert_config_list_to_np


class SurrogateModel(ABC):
    def __init__(self, cs: CS.ConfigurationSpace):
        self.cs = cs

    def __call__(self,
                 X: Union[np.ndarray, CS.Configuration, List[CS.Configuration]]
                 ) -> Union[np.ndarray, float, List[float]]:
        # Config or List[Config] or empty list
        if isinstance(X, CS.Configuration) or (
                isinstance(X, CS.Configuration) and (len(X) == 0) or isinstance(X[0], CS.Configuration)):
            # Returns either float or List[float], depending on whether a single config or list of configs is given
            means, stds = self.predict_configs(X)[0]
        elif isinstance(X, np.ndarray):
            # np.ndarray
            means, stds = self.predict(X)[0]
        else:
            raise TypeError(f"Could not interprete {type(X)}")
        return means

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and sigma of predictions

        :param X: input data shaped [NxK] for N samples and K parameters
        :return: means, stds
        """
        pass

    @abstractmethod
    def fit(self, X: Union[List[CS.Configuration], np.ndarray], y: Union[List[float], np.ndarray]):
        pass

    def predict_configs(self,
                        configs: List[CS.Configuration]) -> Tuple[List[float], List[float]]:
        """
        If configs is a single config: Return a single mean, std.
        If configs is a list of configs: Return a tuple with list of means and list of stds
        """
        X = convert_config_list_to_np(configs)
        y = self.predict(X)
        means = y[0].tolist()
        stds = y[1].tolist()
        return means, stds

    def predict_config(self, config: CS.Configuration) -> Tuple[float, float]:
        # Single config
        mean, std = self.predict(config.get_array())
        assert isinstance(mean, float)
        assert isinstance(std, float)
        return mean, std


class SkLearnPipelineSurrogate(SurrogateModel):
    def __init__(self, pipeline: Pipeline, cs: CS.ConfigurationSpace):
        super().__init__(cs)
        self.pipeline = pipeline

    def fit(self, X: Union[List[CS.Configuration], np.ndarray], y: Union[List[float], np.ndarray]):
        X = convert_config_list_to_np(X)
        self.pipeline.fit(X, y)


class GaussianProcessSurrogate(SkLearnPipelineSurrogate):
    def __init__(self, cs: CS.ConfigurationSpace):
        pipeline = Pipeline([
            ("standardize", StandardScaler()),
            ("GP", GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True,
                                            n_restarts_optimizer=10,
                                            random_state=0)),
        ])
        super().__init__(pipeline, cs)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and sigma of predictions

        :param X: input data shaped [NxK] for N samples and K parameters
        :return: means, sigmas
        """
        return self.pipeline.predict(X, return_std=True)

