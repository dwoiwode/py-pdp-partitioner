import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import matplotlib.pyplot as plt

from src.algorithms.ice import ICE
from src.algorithms.pdp import PDP
from src.demo_data.blackbox_functions import neg_square, square_2D, square
from src.demo_data.config_spaces import config_space_nd
from src.sampler import BayesianOptimizationSampler, RandomSampler
from src.surrogate_models import GaussianProcessSurrogate
from src.utils.plotting import plot_function
from test import PlottableTest


class TestPlottingFunctions(PlottableTest):
    def _apply_blackbox_plot(self, f: callable, cs: CS.ConfigurationSpace, name: str, **kwargs):
        if self.fig is None:
            self.initialize_figure()
        plot_function(f, cs, **kwargs)

        plt.title(name)
        plt.tight_layout()

    def test_plot_square_function_1D(self):
        x = CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)
        self._apply_blackbox_plot(square, cs, "Test Plot Function 1D")

    def test_plot_square_function_1D_low_res(self):
        x = CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        self._apply_blackbox_plot(square, cs, "Test Plot Function 1D (low res)", samples_per_axis=10)

    def test_plot_square_function_2D(self):
        cs = config_space_nd(2)

        self._apply_blackbox_plot(square_2D, cs, "Test Plot Function 2D")

    def test_plot_square_function_2D_low_res(self):
        cs = config_space_nd(2)

        self._apply_blackbox_plot(square_2D, cs, "Test Plot Function 2D (low res)", samples_per_axis=10)

    def test_plot_square_1D_artificial(self):
        cs = config_space_nd(1)

        self._apply_blackbox_plot(square, cs, "Test Plot Function 1D")

        sampler = RandomSampler(square, cs)
        sampler.sample(10)
        sampler.plot(color="red")

    def test_plot_square_2D_artificial(self):
        cs = config_space_nd(2)

        self._apply_blackbox_plot(square_2D, cs, "Test Plot Function 2D")

        sampler = RandomSampler(square_2D, cs)
        sampler.sample(10)
        sampler.plot(color="red")

    def test_plot_square_neg_1D_confidence(self):
        self.initialize_figure()
        cs = config_space_nd(1)

        self._apply_blackbox_plot(neg_square, cs, "Test Plot Confidence 1D, single point")

        bo = BayesianOptimizationSampler(obj_func=neg_square,
                                         config_space=cs,
                                         minimize_objective=False,
                                         initial_points=1)

        bo._sample_initial_points()
        bo.plot()
        bo.surrogate_model.plot(with_confidence=True)
        self.save_fig()
        plt.show()

        for _ in range(20):
            self.initialize_figure()
            self._apply_blackbox_plot(neg_square, cs, "Test Plot Confidence 1D, individual points")

            bo.sample(1)
            bo.surrogate_model.plot(with_confidence=True)
            bo.plot()
            self.save_fig()
            if _ < 3:
                plt.show()

    def test_plot_ice(self):
        self.initialize_figure()
        cs = config_space_nd(2)
        selected_hp = cs.get_hyperparameter("x1")

        sampler = RandomSampler(square_2D, config_space=cs)
        sampler.sample(10)
        sampler.plot(x_hyperparameters=selected_hp)

        surrogate_model = GaussianProcessSurrogate(cs)
        surrogate_model.fit(sampler.X, sampler.y)
        ice = ICE(surrogate_model, selected_hp)
        ice.plot()

    def test_plot_single_ice_with_confidence(self):
        self.initialize_figure()
        cs = config_space_nd(2)
        selected_hp = cs.get_hyperparameter("x1")

        sampler = RandomSampler(square_2D, config_space=cs)
        sampler.sample(10)
        sampler.plot(x_hyperparameters=selected_hp)

        surrogate_model = GaussianProcessSurrogate(cs)
        surrogate_model.fit(sampler.X, sampler.y)
        ice = ICE(surrogate_model, selected_hp)
        ice_curve = ice[0]
        ice_curve.plot(with_confidence=True)

    def test_plot_pdp(self):
        self.initialize_figure()
        cs = config_space_nd(2)
        selected_hp = cs.get_hyperparameter("x1")

        sampler = RandomSampler(square_2D, config_space=cs)
        sampler.sample(10)
        sampler.plot(x_hyperparameters=selected_hp)

        surrogate_model = GaussianProcessSurrogate(cs)
        surrogate_model.fit(sampler.X, sampler.y)
        pdp = PDP(surrogate_model, selected_hp)
        pdp.plot(with_confidence=False)

    def test_plot_pdp_with_confidence(self):
        self.initialize_figure()
        cs = config_space_nd(2)
        selected_hp = cs.get_hyperparameter("x1")

        sampler = RandomSampler(square_2D, config_space=cs)
        sampler.sample(10)
        sampler.plot(x_hyperparameters=selected_hp)

        surrogate_model = GaussianProcessSurrogate(cs)
        surrogate_model.fit(sampler.X, sampler.y)
        pdp = PDP(surrogate_model, selected_hp)
        pdp.plot(with_confidence=True)
