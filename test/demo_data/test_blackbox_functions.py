from unittest import TestCase

import ConfigSpace as CS
import numpy as np
from matplotlib import pyplot as plt

from src.demo_data.blackbox_functions import levy_1D, levy_2D, ackley_1D, ackley_2D, cross_in_tray, styblinski_tang, \
    square_2D, square, neg_square, styblinski_tang_3D_int_1D, styblinski_tang_3D_int_2D, styblinski_tang_integral, \
    styblinski_tang_2D
from src.demo_data.config_spaces import config_space_nd
from src.utils.plotting import plot_function
from test import PlottableTest


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
        for d in range(1, 11):  # Test multiple dimensions
            x = [self.minimum_at] * d
            print(f"Dimensions: {d:2d}, Input: {x}")
            self.assertEqual(len(x), d)
            self.assertAlmostEqual(f(*x), d * self.minimum)

    def test_integral_1d(self):
        f = styblinski_tang
        mean = 0.1 * (styblinski_tang_integral(5) - styblinski_tang_integral(-5))
        f_int = styblinski_tang_3D_int_1D

        x = np.linspace(-5, 5, num=100)
        x = np.expand_dims(x, axis=1)

        for i in range(len(x)):
            f_x = f(x[i])
            f_int_x = f_int(x[i]) - 2 * mean
            self.assertAlmostEqual(f_x, f_int_x, places=5)

    def test_integral_2d(self):
        f = styblinski_tang_2D
        mean = 0.1 * (styblinski_tang_integral(5) - styblinski_tang_integral(-5))
        f_int = styblinski_tang_3D_int_2D

        x1 = np.linspace(-5, 5, num=100)
        x1 = np.expand_dims(x1, axis=1)

        x2 = np.linspace(-5, 5, num=100)
        x2 = np.expand_dims(x2, axis=1)

        for i in range(len(x1)):
            for j in range(len(x2)):
                f_x = f(x1[i], x2[j])
                f_int_x = f_int(x1[i], x2[j]) - mean
                self.assertAlmostEqual(f_x, f_int_x, places=5)


class TestPlotBlackboxFunctions(PlottableTest):
    def _apply_blackbox_plot(self, f: callable, cs: CS.ConfigurationSpace, name: str, **kwargs):
        if self.fig is None:
            self.initialize_figure()
        plot_function(f, cs, **kwargs)

        plt.title(name)
        plt.tight_layout()

    def test_plot_square_1D(self):
        cs = config_space_nd(1)
        self._apply_blackbox_plot(square, cs, "Square 1D")

    def test_plot_neg_square_1D(self):
        cs = config_space_nd(1)
        self._apply_blackbox_plot(neg_square, cs, "Negative Square 1D")

    def test_plot_square_2D(self):
        cs = config_space_nd(2)
        self._apply_blackbox_plot(square_2D, cs, "Square 2D")

    def test_plot_levy_1D(self):
        cs = config_space_nd(1, -10, 10)
        self._apply_blackbox_plot(levy_1D, cs, "Levy 1D")

    def test_plot_levy_2D(self):
        cs = config_space_nd(2, -10, 10)
        self._apply_blackbox_plot(levy_2D, cs, "Levy 2D")

    def test_plot_ackley_1D(self):
        cs = config_space_nd(1, -32.768, 32.768)
        self._apply_blackbox_plot(ackley_1D, cs, "Ackley 1D")

    def test_plot_ackley_2D(self):
        cs = config_space_nd(2, -32.768, 32.768)
        self._apply_blackbox_plot(ackley_2D, cs, "Ackley 2D")

    def test_plot_ackley_1D_zoomed(self):
        cs = config_space_nd(1, -10, 10)
        self._apply_blackbox_plot(ackley_1D, cs, "Ackley 1D")

    def test_plot_ackley_2D_zoomed(self):
        cs = config_space_nd(2, -10, 10)
        self._apply_blackbox_plot(ackley_2D, cs, "Ackley 2D")

    def test_plot_cross_in_tray_2D(self):
        cs = config_space_nd(2, -10, 10)
        self._apply_blackbox_plot(cross_in_tray, cs, "Cross in Tray 2D")

    def test_plot_styblinski_tang_1D(self):
        def styblinski_tang_1D(x):
            return styblinski_tang(x)

        cs = config_space_nd(1)

        self._apply_blackbox_plot(styblinski_tang_1D, cs, "Styblinski Tang 1D")

    def test_plot_styblinski_tang_2D(self):
        def styblinski_tang_2D(x1, x2):
            return styblinski_tang(x1, x2)

        cs = config_space_nd(2)
        self._apply_blackbox_plot(styblinski_tang_2D, cs, "Styblinski Tang 2D")

    def test_plot_styblinski_tang_3D_int_1D(self):
        cs = config_space_nd(1)
        self._apply_blackbox_plot(styblinski_tang_3D_int_1D, cs, "Styblinski Tang Integral 1D")

    def test_plot_styblinski_tang_3D_int_2D(self):
        cs = config_space_nd(2)
        self._apply_blackbox_plot(styblinski_tang_3D_int_2D, cs, "Styblinski Tang Integral 2D")

    def test_plot_styblinski_tang_integral(self):
        def norm_integral(x1: float):
            return 0.1 * styblinski_tang_integral(x1)
        cs = config_space_nd(1)
        self._apply_blackbox_plot(norm_integral, cs, "Styblinski Tang Integral 1D")
