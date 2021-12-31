from typing import Optional

from src.optimizer import AbstractOptimizer, RandomSearch
from src.partitioner import AbstractPartitioner, DecisionTreePartitioner


class PDP:
    def __init__(self,
                 partitioner: Optional[AbstractPartitioner] = None,
                 optimizer: Optional[AbstractOptimizer] = None):
        if partitioner is None:
            partitioner = DecisionTreePartitioner()
        if optimizer is None:
            optimizer = RandomSearch()
        self.partitioner = partitioner
        self.optimizer = optimizer
