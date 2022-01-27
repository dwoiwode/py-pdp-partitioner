from unittest import TestCase

import ConfigSpace as CS
import numpy as np
import pytest
from matplotlib import pyplot as plt

from src.blackbox_functions.synthetic_functions import styblinski_tang_3D_int_1D, styblinski_tang_3D_int_2D, \
    styblinski_tang_integral, Levy, Ackley, CrossInTray, Square, NegativeSquare, StyblinskiTang
from src.blackbox_functions import BlackboxFunction, config_space_nd
from src.utils.plotting import plot_function
from test import PlottableTest


class TestLevy(TestCase):
    def test_levy1D(self):
        f = Levy(1)

        self.assertAlmostEqual(f(x1=1), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            y = f(x1=(np.random.random(1)[0] - 0.5) * 10 * 2)
            self.assertGreater(y, 0)

    def test_levy2D(self):
        f = Levy(2, lower=-10, upper=10)
        self.assertAlmostEqual(f(x1=1, x2=1), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            x1, x2 = (np.random.random(2) - 0.5) * 10 * 2
            y = f(x1=x1, x2=x2)
            self.assertGreater(y, 0)


class TestAckley(TestCase):
    def test_ackley1D(self):
        f = Ackley(1)

        self.assertAlmostEqual(f(x1=0), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            y = f(x1=(np.random.random(1)[0] - 0.5) * 32.768 * 2)
            self.assertGreater(y, 0)

    def test_ackley2D(self):
        f = Ackley(2)
        self.assertAlmostEqual(f(x1=0, x2=0), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            x1, x2 = (np.random.random(2) - 0.5) * 10 * 2
            y = f(x1=x1, x2=x2)
            self.assertGreater(y, 0)


class TestCrossInTray(TestCase):
    def tets_cross_in_tray(self):
        f = CrossInTray()
        # Minima
        self.assertAlmostEqual(f(x1=1.3491, x2=1.3491), -2.06261)
        self.assertAlmostEqual(f(x1=1.3491, x2=-1.3491), -2.06261)
        self.assertAlmostEqual(f(x1=-1.3491, x2=1.3491), -2.06261)
        self.assertAlmostEqual(f(x1=-1.3491, x2=-1.3491), -2.06261)

        # Cannot be smaller than minimum
        for _ in range(10000):
            x1, x2 = (np.random.random(2) - 0.5) * 10 * 2
            y = f(x1=x1, x2=x2)
            self.assertGreater(y, -2.06261)


class TestStyblinskiTang(TestCase):
    minimum = -39.16616570377142
    minimum_at = -2.90353401818596

    def test_styblinski_tang_1D(self):
        f = StyblinskiTang(1)
        self.assertAlmostEqual(f(x1=self.minimum_at), self.minimum)

    def test_minima(self):
        for d in range(1, 11):  # Test multiple dimensions
            f = StyblinskiTang(d)
            x = {f"x{i + 1}": self.minimum_at for i in range(d)}
            print(f"Dimensions: {d:2d}, Input: {x}")
            self.assertEqual(len(x), d)
            self.assertAlmostEqual(f(**x), d * self.minimum)

    def test_integral_1d(self):
        f = StyblinskiTang(1)
        mean = 0.1 * (styblinski_tang_integral(5) - styblinski_tang_integral(-5))
        f_int = styblinski_tang_3D_int_1D

        x = np.linspace(-5, 5, num=100)
        x = np.expand_dims(x, axis=1)

        for i in range(len(x)):
            f_x = f(x1=x[i][0])
            f_int_x = f_int(x[i][0]) - 2 * mean
            self.assertAlmostEqual(f_x, f_int_x, places=5)

    def test_integral_2d(self):
        f = StyblinskiTang.from_n_dimensions(3)
        mean = 0.1 * (styblinski_tang_integral(5) - styblinski_tang_integral(-5))
        f_int = styblinski_tang_3D_int_2D

        for x1 in np.linspace(-5, 5, num=100):
            for x2 in np.linspace(-5, 5, num=100):
                f_x = f(x1=x1, x2=x2)
                f_int_x = f_int(x1, x2) - mean
                self.assertAlmostEqual(f_x, f_int_x, places=5)


class TestPlotBlackboxFunctions(PlottableTest):
    def _apply_blackbox_plot(self, f: callable, cs: CS.ConfigurationSpace, name: str, **kwargs):
        if self.fig is None:
            self.initialize_figure()
        plot_function(f, cs, **kwargs)

        plt.title(name)
        plt.tight_layout()

    def test_plot_ackley_1D_zoomed(self):
        f = Ackley(1, lower=-10, upper=10)
        cs = f.config_space
        self._apply_blackbox_plot(f, cs, "Ackley 1D")

    def test_plot_ackley_2D_zoomed(self):
        f = Ackley(2, lower=-10, upper=10)
        cs = f.config_space
        self._apply_blackbox_plot(f, cs, "Ackley 1D")

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


@pytest.mark.parametrize("f", [
    Square(1), Square(2),
    NegativeSquare(1), NegativeSquare(2),
    Ackley(1), Ackley(2),
    CrossInTray(),
    Levy(1), Levy(2),
    StyblinskiTang(1), StyblinskiTang(2)
])
def test_plot_all(f: BlackboxFunction):
    plt.figure(figsize=(16, 9))
    cs = f.config_space
    plot_function(f, cs)

    plt.title(str(f))
    plt.tight_layout()
    plt.savefig(TestPlotBlackboxFunctions.SAVE_FOLDER / TestPlotBlackboxFunctions.__name__ / f"{str(f)}.png")
    plt.show()
