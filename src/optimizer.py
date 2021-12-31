import abc


class AbstractOptimizer(abc.ABC):
    def __init__(self):
        pass


class GridSearch(AbstractOptimizer):
    pass


class RandomSearch(AbstractOptimizer):
    pass


class BayesianOptiomization(AbstractOptimizer):
    pass
