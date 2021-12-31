from unittest import TestCase

import numpy as np

from src.blackbox_functions import levy1D, levy2D, ackley1D, ackley2D, cross_in_tray

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from .test_plotting import TestPlotting


class TestLevy(TestCase):
    def test_levy1D(self):
        f = levy1D

        self.assertAlmostEqual(f(1), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            y = f(*((np.random.random(1) - 0.5) * 10 * 2))
            self.assertGreater(y, 0)

    def test_levy2D(self):
        f = levy2D
        self.assertAlmostEqual(f(1, 1), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            y = f(*((np.random.random(2) - 0.5) * 10 * 2))
            self.assertGreater(y, 0)


class TestAckley(TestCase):
    def test_ackley1D(self):
        f = ackley1D

        self.assertAlmostEqual(f(0), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            y = f(*((np.random.random(1) - 0.5) * 32.768 * 2))
            self.assertGreater(y, 0)

    def test_ackley2D(self):
        f = ackley2D
        self.assertAlmostEqual(f(0, 0), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            y = f(*((np.random.random(2) - 0.5) * 32.768 * 2))
            self.assertGreater(y, 0)

class TestCrossInTray(TestCase):
    def tets_cross_in_tray(self):
        f = cross_in_tray
        # Minima
        self.assertAlmostEqual(f(1.3491, 1.3491), -2.06261)
        self.assertAlmostEqual(f(1.3491, -1.3491), -2.06261)
        self.assertAlmostEqual(f(-1.3491, 1.3491), -2.06261)
        self.assertAlmostEqual(f(-1.3491, -1.3491), -2.06261)

        # Cannot be smaller than minimum
        for _ in range(10000):
            y = f(*((np.random.random(2) - 0.5) * 10 * 2))
            self.assertGreater(y, -2.06261)

class TestPlotBlackboxFunctions(TestPlotting):
    def test_plot_levy_1D(self):
        hps = [
            CSH.UniformFloatHyperparameter("x", lower=-10, upper=10),
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(levy1D, cs, "Levy 1D")

    def test_plot_levy_2D(self):
        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(levy2D, cs, "Levy 2D")

    def test_plot_ackley_1D(self):
        hps = [
            CSH.UniformFloatHyperparameter("x", lower=-32.768, upper=32.768),
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(ackley1D, cs, "Ackley 1D")

    def test_plot_ackley_2D(self):
        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-32.768, upper=32.768),
            CSH.UniformFloatHyperparameter("x2", lower=-32.768, upper=32.768)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(ackley2D, cs, "Ackley 2D")

    def test_plot_ackley_1D_zoomed(self):
        hps = [
            CSH.UniformFloatHyperparameter("x", lower=-10, upper=10),
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(ackley1D, cs, "Ackley 1D")

    def test_plot_ackley_2D_zoomed(self):
        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(ackley2D, cs, "Ackley 2D")

    def test_plot_cross_in_tray_2D(self):
        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(cross_in_tray, cs, "Cross in Tray 2D")
