from unittest import TestCase

import numpy as np

from src.blackbox_functions import levy_1D, levy_2D, ackley_1D, ackley_2D, cross_in_tray, styblinski_tang

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from .test_plotting import TestPlotting


class TestLevy(TestCase):
    def test_levy1D(self):
        f = levy_1D

        self.assertAlmostEqual(f(1), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            y = f(*((np.random.random(1) - 0.5) * 10 * 2))
            self.assertGreater(y, 0)

    def test_levy2D(self):
        f = levy_2D
        self.assertAlmostEqual(f(1, 1), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            y = f(*((np.random.random(2) - 0.5) * 10 * 2))
            self.assertGreater(y, 0)


class TestAckley(TestCase):
    def test_ackley1D(self):
        f = ackley_1D

        self.assertAlmostEqual(f(0), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            y = f(*((np.random.random(1) - 0.5) * 32.768 * 2))
            self.assertGreater(y, 0)

    def test_ackley2D(self):
        f = ackley_2D
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


class TestStyblinskiTang(TestCase):
    minimum = -39.16616570377142
    minimum_at = -2.90353401818596
    def test_styblinski_tang_1D(self):
        f = styblinski_tang
        self.assertAlmostEqual(f(self.minimum_at), self.minimum)

    def test_minima(self):
        f = styblinski_tang
        for d in range(1, 11):
            x = [self.minimum_at] * d
            print(f"Dimensions: {d:2d}, Input: {x}")
            self.assertEqual(len(x), d)
            self.assertAlmostEqual(f(*x), d * self.minimum)


class TestPlotBlackboxFunctions(TestPlotting):
    def test_plot_levy_1D(self):
        hps = [
            CSH.UniformFloatHyperparameter("x", lower=-10, upper=10),
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(levy_1D, cs, "Levy 1D")

    def test_plot_levy_2D(self):
        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(levy_2D, cs, "Levy 2D")

    def test_plot_ackley_1D(self):
        hps = [
            CSH.UniformFloatHyperparameter("x", lower=-32.768, upper=32.768),
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(ackley_1D, cs, "Ackley 1D")

    def test_plot_ackley_2D(self):
        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-32.768, upper=32.768),
            CSH.UniformFloatHyperparameter("x2", lower=-32.768, upper=32.768)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(ackley_2D, cs, "Ackley 2D")

    def test_plot_ackley_1D_zoomed(self):
        hps = [
            CSH.UniformFloatHyperparameter("x", lower=-10, upper=10),
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(ackley_1D, cs, "Ackley 1D")

    def test_plot_ackley_2D_zoomed(self):
        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(ackley_2D, cs, "Ackley 2D")

    def test_plot_cross_in_tray_2D(self):
        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-10, upper=10),
            CSH.UniformFloatHyperparameter("x2", lower=-10, upper=10)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        self._apply_blackbox_plot(cross_in_tray, cs, "Cross in Tray 2D")

    def test_plot_styblinski_tang_1D(self):
        def styblinski_tang_1D(x):
            return styblinski_tang(x)

        hps = [
            CSH.UniformFloatHyperparameter("x", lower=-5, upper=5),
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)

        self._apply_blackbox_plot(styblinski_tang_1D, cs, "Styblinski Tang 1D")

    def test_plot_styblinski_tang_2D(self):
        def styblinski_tang_2D(x1, x2):
            return styblinski_tang(x1, x2)

        hps = [
            CSH.UniformFloatHyperparameter("x1", lower=-5, upper=5),
            CSH.UniformFloatHyperparameter("x2", lower=-5, upper=5)
        ]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)

        self._apply_blackbox_plot(styblinski_tang_2D, cs, "Styblinski Tang 2D")
