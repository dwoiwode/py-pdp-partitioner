"""
Overview over possible theoretically possible benchmarks:
https://github.com/automl/HPOBench/wiki/Available-Containerized-Benchmarks
"""
from typing import Callable, Any, Tuple, Union

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient
import ConfigSpace as CS

from pyPDP.blackbox_functions import BlackboxFunction

"""
Viable OpenML Task IDs:
2079: 2D Configspace
"""


class HPOBenchBlackbox(BlackboxFunction):
    def __init__(self, benchmark: Union[AbstractBenchmark, AbstractBenchmarkClient], *, seed=None):
        self.benchmark = benchmark
        super(HPOBenchBlackbox, self).__init__(benchmark.get_configuration_space(seed))
        self.__seed = seed

    def __repr__(self) -> str:
        return f"HPOBenchmark(benchmark={self.benchmark.__class__.__name__})"

    def value_from_config(self, config: CS.Configuration) -> float:
        result_dict = self.benchmark.objective_function(configuration=config, rng=self.__seed)
        return result_dict["function_value"]


def get_Cifar100NasBench201Benchmark(seed=0) -> Tuple[CS.ConfigurationSpace, Callable[[Any], float]]:
    """
    Takes a lot of memory and some time to load...

    Configspace consists of 6 categorical hyperparameter consisting of 5 options:
    - ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    """
    from hpobench.benchmarks.nas.nasbench_201 import Cifar100NasBench201Benchmark
    b = Cifar100NasBench201Benchmark(rng=seed)
    f = HPOBenchBlackbox(b, seed=seed)
    cs = f.config_space

    return cs, f


def get_BNNOnBostonHousing(seed=0):
    """
    Requires installed package openml
    """
    from hpobench.benchmarks.ml.pybnn import BNNOnBostonHousing
    b = BNNOnBostonHousing(rng=seed)
    f = HPOBenchBlackbox(b, seed=seed)
    cs = f.config_space

    return cs, f


def get_Paramnet(seed=0):
    """
    """
    from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamNetAdultOnTimeBenchmark
    b = ParamNetAdultOnTimeBenchmark(rng=seed)
    f = HPOBenchBlackbox(b, seed=seed)
    cs = f.config_space

    return cs, f


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

    f = HPOBenchBlackbox(b, seed=seed)
    cs = f.config_space

    return cs, f


def get_SVMBenchmarkMF(task_id: int = 2079, seed=0):
    """
    task_id: see https://openml.github.io/openml-python/develop/examples/30_extended/tasks_tutorial.html
    Requires openml
    """
    from hpobench.benchmarks.ml import SVMBenchmarkMF
    b = SVMBenchmarkMF(task_id, rng=seed)

    f = HPOBenchBlackbox(b, seed=seed)
    cs = f.config_space

    return cs, f


def get_RFBenchmarkMF(task_id: int = 2079, seed=0):
    """
    task_id: see https://openml.github.io/openml-python/develop/examples/30_extended/tasks_tutorial.html
    Requires openml
    """
    from hpobench.benchmarks.ml import RandomForestBenchmarkMF
    b = RandomForestBenchmarkMF(task_id, rng=seed)

    f = HPOBenchBlackbox(b, seed=seed)
    cs = f.config_space

    return cs, f


def get_NNBenchmarkMF(task_id: int = 2079, seed=0):
    """
    task_id: see https://openml.github.io/openml-python/develop/examples/30_extended/tasks_tutorial.html
    Requires openml
    """
    from hpobench.benchmarks.ml import NNBenchmarkMF
    b = NNBenchmarkMF(task_id, rng=seed)

    f = HPOBenchBlackbox(b, seed=seed)
    cs = f.config_space

    return cs, f
