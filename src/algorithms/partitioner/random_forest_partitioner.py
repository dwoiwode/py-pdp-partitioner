from typing import Optional, Iterable, List

from ConfigSpace import hyperparameters as CSH
from matplotlib import pyplot as plt

from src.algorithms.partitioner import Partitioner, Region


class RandomForestPartitioner(Partitioner):
    def partition(self, max_depth: int = 1) -> List[Region]:
        pass

    def plot(self, *args, x_hyperparameters: Optional[Iterable[CSH.Hyperparameter]] = None,
             ax: Optional[plt.Axes] = None):
        pass
