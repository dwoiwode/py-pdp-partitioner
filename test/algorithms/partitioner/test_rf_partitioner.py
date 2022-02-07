from src.algorithms.ice import ICE
from src.algorithms.partitioner.random_forest_partitioner import RandomForestPartitioner
from src.blackbox_functions.synthetic_functions import Square, StyblinskiTang
from src.sampler.bayesian_optimization import BayesianOptimizationSampler
from src.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from test import PlottableTest


class TestRFPartitioner(PlottableTest):
    def test_rf_simple(self):
        self.initialize_figure()
        f = Square.for_n_dimensions(2)
        cs = f.config_space
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(f, config_space=cs)
        bo.sample(50)

        surrogate = GaussianProcessSurrogate(cs)
        surrogate.fit(bo.X, bo.y)

        ice = ICE.from_random_points(surrogate, selected_hp)
        partitioner = RandomForestPartitioner.from_ICE(ice)
        partitioner.partition(num_trees=10, max_depth=1, sample_size=20)

        incumbent_region = partitioner.get_incumbent_region(bo.incumbent_config)
        self.assertLess(0, len(incumbent_region))

        partitioner.plot_incumbent_regions(bo.incumbent_config)
        self.save_fig()

    def test_rf_3D_single_split(self):
        self.initialize_figure()
        f = StyblinskiTang.for_n_dimensions(3, seed=32)
        cs = f.config_space
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(f, config_space=cs)
        bo.sample(80)

        surrogate = GaussianProcessSurrogate(cs)
        surrogate.fit(bo.X, bo.y)

        ice = ICE.from_random_points(surrogate, selected_hp)
        partitioner = RandomForestPartitioner.from_ICE(ice)
        partitioner.partition(num_trees=20, max_depth=1, sample_size=100)

        incumbent_region = partitioner.get_incumbent_region(bo.incumbent_config)
        self.assertLess(0, len(incumbent_region))

        partitioner.plot_incumbent_regions(bo.incumbent_config)
        self.save_fig()

    def test_rf_3D_two_splits(self):
        self.initialize_figure()
        f = StyblinskiTang.for_n_dimensions(3, seed=32)
        cs = f.config_space
        selected_hp = cs.get_hyperparameter("x1")

        bo = BayesianOptimizationSampler(f, config_space=cs)
        bo.sample(80)

        surrogate = GaussianProcessSurrogate(cs)
        surrogate.fit(bo.X, bo.y)

        ice = ICE.from_random_points(surrogate, selected_hp)
        partitioner = RandomForestPartitioner.from_ICE(ice)
        partitioner.partition(num_trees=30, max_depth=2, sample_size=200)

        incumbent_region = partitioner.get_incumbent_region(bo.incumbent_config, min_incumbent_overlap=5)
        self.assertLess(0, len(incumbent_region))

        partitioner.plot_incumbent_regions(bo.incumbent_config)
        self.save_fig()
