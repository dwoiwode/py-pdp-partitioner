from pathlib import Path
from unittest import TestCase

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import matplotlib.pyplot as plt

from src.blackbox_functions import neg_square, square_2D, square
from src.config_spaces import config_space_nd
from src.optimizer import BayesianOptimization
from src.pdp import PDP
from src.plotting import plot_function, plot_model_confidence, plot_samples, style_axes, finalize_figure, plot_ice, \
    plot_pdp, plot_confidence_lists
from src.utils import unscale


class TestPlotting(TestCase):
    SHOW = True
    SAVE_FOLDER = Path(__file__).parent / "plots"

    @classmethod
    def setUpClass(cls) -> None:
        if cls.SAVE_FOLDER is not None:
            cls.SAVE_FOLDER.mkdir(exist_ok=True, parents=True)

    def setUp(self) -> None:
        self.no_plot = False
        # Make sure that figure is cleared from previous tests
        plt.clf()

    def tearDown(self) -> None:
        # Save figure from last test
        if self.no_plot:
            return
        fig = plt.gcf()
        finalize_figure(fig)
        if self.SAVE_FOLDER is not None:
            plt.savefig(self.SAVE_FOLDER / f"{self.__class__.__name__}_{self._testMethodName}.png")

        # Show plot from last test
        if self.SHOW:
            plt.show()
        plt.clf()

    def _apply_blackbox_plot(self, f: callable, cs: CS.ConfigurationSpace, name: str, **kwargs):
        fig = plt.figure()
        ax = fig.gca()
        ax = plot_function(f, cs, **kwargs, ax=ax)
        self.assertIsInstance(ax, plt.Axes)

        plt.title(name)
        style_axes(ax, cs)
        plt.tight_layout()
        return fig


class TestPlottingFunctions(TestPlotting):
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
        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)

        self._apply_blackbox_plot(square_2D, cs, "Test Plot Function 2D")

    def test_plot_square_function_2D_low_res(self):
        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)

        self._apply_blackbox_plot(square_2D, cs, "Test Plot Function 2D (low res)", samples_per_axis=10)

    def test_plot_square_1D_artificial(self):
        x = CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        samples = cs.sample_configuration(10)

        fig = self._apply_blackbox_plot(square, cs, "Test Plot Function 1D")
        ax = fig.gca()
        plot_samples(samples, [square(**sample) for sample in samples], ax=ax)

    def test_plot_square_2D_artificial(self):
        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)

        samples = cs.sample_configuration(10)

        fig = self._apply_blackbox_plot(square_2D, cs, "Test Plot Function 2D")
        ax = fig.gca()
        plot_samples(samples, [square_2D(**sample) for sample in samples], ax=ax)

    def test_plot_square_neg_1D_confidence(self):
        x = CSH.UniformFloatHyperparameter("x1", lower=-1, upper=1)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        bo = BayesianOptimization(obj_func=neg_square, config_space=cs, minimize_objective=False,
                                  initial_points=1, eps=0.1)

        res = bo.optimize(0)
        fig = self._apply_blackbox_plot(neg_square, cs, "Test Plot Confidence 1D, single point")
        ax = fig.gca()
        plot_samples(bo.config_list, bo.y_list, ax=ax)
        plot_model_confidence(bo, cs, ax=ax)
        finalize_figure(fig)
        fig.show()

        for _ in range(2):
            bo.optimize(1)
            fig2 = self._apply_blackbox_plot(neg_square, cs, "Test Plot Confidence 1D, two points")
            ax = fig2.gca()
            plot_samples(bo.config_list, bo.y_list, ax=ax)
            plot_model_confidence(bo, cs, ax=ax)
            finalize_figure(fig2)
            fig2.show()

        bo.optimize(20)
        fig3 = self._apply_blackbox_plot(neg_square, cs, "Test Plot Confidence 1D, two points")
        ax = fig3.gca()
        plot_samples(bo.config_list, bo.y_list, ax=ax)
        plot_model_confidence(bo, cs, ax=ax)

    def test_plot_ice(self):
        cs = config_space_nd(2)
        bo = BayesianOptimization(square_2D, config_space=cs)
        selected_hp = cs.get_hyperparameters()[0]
        bo.optimize(10)
        pdp = PDP(bo.surrogate_model,  cs)
        idx = 0
        x_ice, y_ice, variances = pdp.calculate_ice(selected_hp, centered=False)
        x_unscaled = unscale(x_ice, cs)
        fig = self._apply_blackbox_plot(square, config_space_nd(1), "Test Plot ICE")
        ax = fig.gca()
        plot_ice(x_unscaled, y_ice, idx, ax=ax)
        finalize_figure(fig)

    def test_plot_single_ice_with_confidence(self):
        cs = config_space_nd(2)
        bo = BayesianOptimization(square_2D, config_space=cs)
        selected_hp = cs.get_hyperparameters()[0]
        bo.optimize(10)
        pdp = PDP(bo.surrogate_model,  cs)
        idx = 0
        x_ice, y_ice, variances = pdp.calculate_ice(selected_hp, centered=False)
        curve_idx = 0
        x_single = x_ice[curve_idx]
        y_single = y_ice[curve_idx]
        variance_single = variances[curve_idx]
        x_unscaled = unscale(x_single, cs)
        fig = self._apply_blackbox_plot(square, config_space_nd(1), "Test Plot Single ICE Curve")
        ax = fig.gca()
        plot_confidence_lists(x_unscaled[:, idx], y_single, variances=variance_single, ax=ax)
        finalize_figure(fig)


    def test_plot_pdp(self):
        cs = config_space_nd(2)
        bo = BayesianOptimization(square_2D, config_space=cs)
        selected_hp = cs.get_hyperparameters()[0]
        bo.optimize(10)
        pdp = PDP(bo.surrogate_model,  cs)
        idx = 0

        # x_ice, y_ice, variances = pdp.calculate_ice(idx, centered=False)
        # x_unscaled_ice = unscale(x_ice, config_space_nd(2))

        x_pdp, y_pdp, variances = pdp.calculate_pdp(selected_hp, centered=False)
        x_unscaled_pdp = unscale(x_pdp, cs)

        fig = self._apply_blackbox_plot(square, config_space_nd(1), "Test Plot PDP")
        ax = fig.gca()
        plot_confidence_lists(x_unscaled_pdp[:, idx], y_pdp, variances=variances, ax=ax)
        # plot_ice(x_unscaled_ice, y_ice, idx, ax=ax)
        plot_pdp(x_unscaled_pdp, y_pdp, idx, ax=ax)
        finalize_figure(fig)
