import numpy as np
from matplotlib import pyplot as plt

from src.algorithms.ice import ICE
from src.algorithms.partitioner.decision_tree_partitioner import DTPartitioner
from src.demo_data.blackbox_functions import square_2D
from src.demo_data.config_spaces import config_space_nd
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from test import PlottableTest


class TestPartitioner(PlottableTest):
    def test_dt_partitioner_single_split(self):
        self.initialize_figure()
        cs = config_space_nd(2)
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(square_2D, config_space=cs)
        bo.sample(10)

        ice = ICE(bo.surrogate_model,  selected_hp)
        partitioner = DTPartitioner.from_ICE(ice)
        regions = partitioner.partition()

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
        self.save_fig()
        plt.show()

    def test_dt_partitioner_multiple_splits(self):
        cs = config_space_nd(2, lower=-1, upper=1)
        selected_hp = cs.get_hyperparameters()[0]

        bo = BayesianOptimizationSampler(square_2D, config_space=cs)
        bo.sample(10)
        ice = ICE(bo.surrogate_model,  selected_hp)

        partitioner = DTPartitioner.from_ICE(ice)
        regions = partitioner.partition(max_depth=3)

        num_points = 0
        for region in regions:
            num_points += len(region)
        self.assertEqual(ice.num_samples, num_points)

        colors = ['red', 'orange', 'green', 'blue', 'grey', 'black', 'magenta', 'yellow']
        color_list = colors[:len(regions)]
        partitioner.plot(color_list=color_list)
        # self.save_fig()
        plt.show()
