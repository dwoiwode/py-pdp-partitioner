import numpy as np
from matplotlib import pyplot as plt

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from src.blackbox_functions.synthetic_functions import Square
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from test import PlottableTest


class TestPartitioner(PlottableTest):
    def test_dt_partitioner_single_split(self):
        self.initialize_figure()
        f = Square.for_n_dimensions(2)
        cs = f.config_space
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(f, config_space=cs)
        bo.sample(10)

        surrogate = GaussianProcessSurrogate(cs)
        surrogate.fit(bo.X, bo.y)

        ice = ICE.from_random_points(surrogate, selected_hp)
        partitioner = DecisionTreePartitioner.from_ICE(ice)
        # regions = partitioner.partition()
        partitioner.partition()
        regions = partitioner.leaves

        num_points = 0
        for region in regions:
            num_points += len(region)
        self.assertEqual(ice.num_samples, num_points)

        mean_confidences = np.asarray([region.mean_confidence for region in regions])
        self.assertTrue(np.all(np.abs(mean_confidences)))

        # plotting in different colors
        colors = ['green', 'blue']
        for i, region in enumerate(regions):
            region.plot(colors[i])

    def test_dt_partitioner_multiple_splits(self):
        self.initialize_figure()
        f = Square.for_n_dimensions(2, lower=-1, upper=1)
        cs = f.config_space
        selected_hp = cs.get_hyperparameters()[0]

        bo = BayesianOptimizationSampler(f, config_space=cs)
        bo.sample(10)

        surrogate = GaussianProcessSurrogate(cs)
        surrogate.fit(bo.X, bo.y)

        ice = ICE.from_random_points(surrogate,  selected_hp)

        partitioner = DecisionTreePartitioner.from_ICE(ice)
        regions = partitioner.partition(max_depth=3)

        num_points = 0
        for region in regions:
            num_points += len(region)
        self.assertEqual(ice.num_samples, num_points)

        colors = ['red', 'orange', 'green', 'blue', 'grey', 'black', 'magenta', 'yellow']
        color_list = colors[:len(regions)]
        partitioner.plot(color_list=color_list)
