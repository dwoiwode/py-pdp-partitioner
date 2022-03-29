from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from tests.surrogates import SurrogateTest


class TestGaussianProcess(SurrogateTest):
    surrogate_class = GaussianProcessSurrogate
