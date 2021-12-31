import abc


class AbstractPartitioner(abc.ABC):
    def __init__(self):
        pass


class DecisionTreePartitioner(AbstractPartitioner):
    pass


class RandomForestPartitioner(AbstractPartitioner):
    pass
