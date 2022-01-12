import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def square_config_space() -> CS.ConfigurationSpace:
    x = CSH.UniformFloatHyperparameter("x", lower=-4, upper=4)
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(x)
    return cs

def square_2D_config_space() -> CS.ConfigurationSpace:
    x1 = CSH.UniformFloatHyperparameter("x1", lower=-4, upper=4)
    x2 = CSH.UniformFloatHyperparameter("x2", lower=-4, upper=4)
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(x1)
    cs.add_hyperparameter(x2)
    return cs