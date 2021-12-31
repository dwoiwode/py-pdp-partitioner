from pathlib import Path
from unittest import TestCase

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import matplotlib.pyplot as plt

from src.utils import plot_function


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
