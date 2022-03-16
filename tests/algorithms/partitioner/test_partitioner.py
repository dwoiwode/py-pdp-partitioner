import matplotlib
import numpy as np

from pyPDP.algorithms.ice import ICE
from pyPDP.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from pyPDP.blackbox_functions.synthetic_functions import Square, StyblinskiTang
from pyPDP.sampler.bayesian_optimization import BayesianOptimizationSampler
from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from pyPDP.utils.plotting import plot_function, plot_config_space
from tests import PlottableTest


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
        f = StyblinskiTang.for_n_dimensions(2)
        cs = f.config_space
        selected_hp = "x1"

        bo = BayesianOptimizationSampler(f, config_space=cs)
        bo.sample(80)

        surrogate = GaussianProcessSurrogate(cs)
        surrogate.fit(bo.X, bo.y)

        ice = ICE.from_random_points(surrogate, selected_hp, num_grid_points_per_axis=100)

        partitioner = DecisionTreePartitioner.from_ICE(ice)
        regions = partitioner.partition(max_depth=3)

        num_points = 0
        for region in regions:
            num_points += len(region)
        self.assertEqual(ice.num_samples, num_points)

        colors = ['red', 'orange', 'green', 'blue', 'grey', 'black', 'magenta', 'yellow']
        color_list = colors[:len(regions)]
        partitioner.plot(color_list=color_list)

    def test_dt_partitioner_multiple_splits_3d(self):
        self.initialize_figure()
        f = StyblinskiTang.for_n_dimensions(3)
        cs = f.config_space
        selected_hp = "x3"

        bo = BayesianOptimizationSampler(f, config_space=cs, seed=0)
        bo.sample(150)

        surrogate = GaussianProcessSurrogate(cs, seed=0)
        surrogate.fit(bo.X, bo.y)

        ice = ICE.from_random_points(surrogate, selected_hp, num_grid_points_per_axis=100)

        partitioner = DecisionTreePartitioner.from_ICE(ice)
        partitioner.partition(max_depth=4)

        ax = self.fig.gca()
        f_2d = StyblinskiTang.for_n_dimensions(2, seed=0)
        plot_function(f_2d, f_2d.config_space, ax=ax)

        RANDOM_COLORS = tuple(matplotlib.colors.BASE_COLORS.values())
        for i, leaf in enumerate(partitioner.leaves):
            plot_config_space(leaf.implied_config_space(seed=0), color=RANDOM_COLORS[i % len(RANDOM_COLORS)], alpha=0.3,
                              x_hyperparameters=("x1", "x2"))
