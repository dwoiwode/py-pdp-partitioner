"""
Overview over possible theoretically possible benchmarks:
https://github.com/automl/HPOBench/wiki/Available-Containerized-Benchmarks
"""
from typing import Callable, Any, Tuple, Union

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient
import ConfigSpace as CS

"""
Viable OpenML Task IDs:
2079: 2D Configspace
"""


def _hpo_blackbox_function(benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient], seed=0) -> Callable[[Any], float]:
    """
    Converts an objective function that takes a config as input in to a blackbox function
    that takes keyword arguments as input
    :param benchmark: Used benchmark
    :param seed: Seed for experiment
    """
    cs = benchmark.get_configuration_space(seed=seed)

    def bb_function(**kwargs):
        config = CS.Configuration(cs, values=kwargs)
        result_dict = benchmark.objective_function(configuration=config, rng=seed)
        return result_dict["function_value"]

    bb_function.__name__ = benchmark.__class__.__name__
    return bb_function


def get_Cifar100NasBench201Benchmark(seed=0) -> Tuple[CS.ConfigurationSpace, Callable[[Any], float]]:
    """
    Takes a lot of memory and some time to load...

    Configspace consists of 6 categorical hyperparameter consisting of 5 options:
    - ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    """
    from hpobench.benchmarks.nas.nasbench_201 import Cifar100NasBench201Benchmark
    b = Cifar100NasBench201Benchmark(rng=seed)
    cs = b.get_configuration_space(seed)

    return cs, _hpo_blackbox_function(b, seed)


def get_BNNOnBostonHousing(seed=0):
    """
    Requires installed package openml
    """
    from hpobench.benchmarks.ml.pybnn import BNNOnBostonHousing
    b = BNNOnBostonHousing(rng=seed)
    cs = b.get_configuration_space(seed)

    return cs, _hpo_blackbox_function(b, seed)


def get_Paramnet(seed=0):
    """
    """
    from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamNetAdultOnTimeBenchmark
    b = ParamNetAdultOnTimeBenchmark(rng=seed)
    cs = b.get_configuration_space(seed)

    return cs, _hpo_blackbox_function(b, seed)


def get_Cartpole(seed=0, reduced=True):
    """
    Requires tensorflow, gym, tensorforce
    """
    if reduced:
        from hpobench.benchmarks.rl.cartpole import CartpoleReduced
        b = CartpoleReduced(rng=seed)
    else:
        from hpobench.benchmarks.rl.cartpole import CartpoleFull
        b = CartpoleFull(rng=seed)

    cs = b.get_configuration_space(seed)

    return cs, _hpo_blackbox_function(b, seed)


def get_SVMBenchmarkMF(task_id:int=2079, seed=0):
    """
    task_id: see https://openml.github.io/openml-python/develop/examples/30_extended/tasks_tutorial.html
    Requires openml
    """
    from hpobench.benchmarks.ml import SVMBenchmarkMF
    b = SVMBenchmarkMF(task_id, rng=seed)

    cs = b.get_configuration_space(seed)

    return cs, _hpo_blackbox_function(b, seed)
