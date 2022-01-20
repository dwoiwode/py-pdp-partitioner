import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def config_space_nd(dimensions, lower=-5, upper=5, seed=None) -> CS.ConfigurationSpace:
    cs = CS.ConfigurationSpace(seed=seed)
    for i in range(dimensions):
        x = CSH.UniformFloatHyperparameter(f"x{i + 1}", lower=lower, upper=upper)
        cs.add_hyperparameter(x)
    return cs
