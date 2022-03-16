import ConfigSpace as CS
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyPDP.surrogate_models import SurrogateModel


class SkLearnPipelineSurrogate(SurrogateModel):
    def __init__(self, pipeline: Pipeline, cs: CS.ConfigurationSpace, seed=None):
        super().__init__(cs, seed=seed)
        self.pipeline = pipeline

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and sigma of predictions

        :param X: input data shaped [NxK] for N samples and K parameters
        :return: means, sigmas
        """

        return self.pipeline.predict(X, return_std=True)

class GaussianProcessSurrogate(SkLearnPipelineSurrogate):
    def __init__(self, cs: CS.ConfigurationSpace, kernel=Matern(nu=1.5), seed=None):
        pipeline = Pipeline([
            ("standardize", StandardScaler()),
            ("GP", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                            n_restarts_optimizer=20,
                                            alpha=1e-8,
                                            random_state=seed)),
        ])
        super().__init__(pipeline, cs, seed=seed)
