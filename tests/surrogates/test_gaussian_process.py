from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from tests.surrogates import AbstractTestSurrogate


class TestGaussianProcess(AbstractTestSurrogate):
    surrogate_class = GaussianProcessSurrogate
