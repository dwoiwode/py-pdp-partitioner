from pathlib import Path
from unittest import TestCase

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import matplotlib.pyplot as plt

from src.optimizer import BayesianOptimization
from src.utils import plot_function, config_to_array


class TestPlotting(TestCase):
    SHOW = True
    SAVE_FOLDER = Path(__file__).parent / "plots"

    @classmethod
    def setUpClass(cls) -> None:
        if cls.SAVE_FOLDER is not None:
            cls.SAVE_FOLDER.mkdir(exist_ok=True, parents=True)

    def setUp(self) -> None:
        # Make sure that figure is cleared from previous tests
        plt.clf()

    def tearDown(self) -> None:
        # Save figure from last test
        if self.SAVE_FOLDER is not None:
            plt.savefig(self.SAVE_FOLDER / f"{self.__class__.__name__}_{self._testMethodName}.png")

        # Show plot from last test
        if self.SHOW:
            plt.show()
        plt.clf()

    def _apply_blackbox_plot(self, f: callable, cs: CS.ConfigurationSpace, name: str, **kwargs):
        fig = plot_function(f, cs, **kwargs)
        self.assertIsInstance(fig, plt.Figure)

        plt.title(name)
        plt.tight_layout()
        return fig


class TestPlottingFunctions(TestPlotting):
    def test_plot_square_function_1D(self):
        def square(x):
            return x ** 2

        x = CSH.UniformFloatHyperparameter("x", lower=-10, upper=10)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)
        self._apply_blackbox_plot(square, cs, "Test Plot Function 1D")

    def test_plot_square_function_1D_low_res(self):
        def square(x):
            return x ** 2

        x = CSH.UniformFloatHyperparameter("x", lower=-10, upper=10)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        self._apply_blackbox_plot(square, cs, "Test Plot Function 1D (low res)", samples_per_axis=10)

    def test_plot_square_function_2D(self):
        def square2D(x1, x2):
            return x1 ** 2 + x2 ** 2

        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)

        self._apply_blackbox_plot(square2D, cs, "Test Plot Function 2D")

    def test_plot_square_function_2D_low_res(self):
        def square2D(x1, x2):
            return x1 ** 2 + x2 ** 2

        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)

        self._apply_blackbox_plot(square2D, cs, "Test Plot Function 2D (low res)", samples_per_axis=10)

    def test_plot_square_1D_artificial(self):
        def square(x):
            return x ** 2

        x = CSH.UniformFloatHyperparameter("x", lower=-10, upper=10)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        samples = cs.sample_configuration(10)

        self._apply_blackbox_plot(square, cs, "Test Plot Function 1D", config_samples=samples)

    def test_plot_square_2D_artificial(self):
        def square2D(x1, x2):
            return x1 ** 2 + x2 ** 2

        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)

        samples = cs.sample_configuration(10)

        self._apply_blackbox_plot(square2D, cs, "Test Plot Function 1D", config_samples=samples)

    def test_plot_square_neg_1D_confidence(self):
        def neg_square(x):
            return 1 - x ** 2

        def neg_square_with_cs(config: CS.Configuration):
            arr = config_to_array(config)
            return 1 - arr[0] ** 2

        x = CSH.UniformFloatHyperparameter("x", lower=-1, upper=1)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        bo = BayesianOptimization(obj_func=neg_square_with_cs, config_space=cs, minimize_objective=False,
                                  initial_points=1, eps=0.1)

        res = bo.sample_new_points(0)
        fig = self._apply_blackbox_plot(neg_square, cs, "Test Plot Confidence 1D, single point",
                                        config_samples=bo.config_list, model=bo.surrogate_score)
        fig.show()

        for _ in range(2):
            bo.sample_new_points(1)
            fig2 = self._apply_blackbox_plot(neg_square, cs, "Test Plot Confidence 1D, two points",
                                             config_samples=bo.config_list, model=bo.surrogate_score)
            fig2.show()

        bo.sample_new_points(20)
        fig3 = self._apply_blackbox_plot(neg_square, cs, "Test Plot Confidence 1D, two points",
                                         config_samples=bo.config_list, model=bo.surrogate_score)
        fig3.show()







