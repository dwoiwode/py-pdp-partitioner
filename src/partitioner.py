import abc
from typing import List, Tuple

import ConfigSpace as CS
import numpy as np

Sample = Tuple[np.ndarray, np.ndarray]  # configurations, variances

class AbstractPartitioner(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def partition(self, configuraitons: np.ndarray, variances: np.ndarray, depth=1):
        pass

    def split(self, samples: Sample, j: int, t: int) -> Tuple[Sample, Sample]:
        config_array, variances = samples
        splitting_on = config_array[t, j]

        n1 = splitting_on


class DecisionTreePartitioner(AbstractPartitioner):
    def partition(self, samples: List[Tuple[CS.Configuration,]]):
        pass


class RandomForestPartitioner(AbstractPartitioner):
    pass
