from unittest import TestCase

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import pytest
from matplotlib import pyplot as plt

from pyPDP.blackbox_functions import BlackboxFunction, config_space_nd
from pyPDP.blackbox_functions.synthetic_functions import Levy, Ackley, CrossInTray, Square, NegativeSquare, \
    StyblinskiTang
from pyPDP.utils.plotting import plot_function
from tests import PlottableTest


class TestConfigspaceND(TestCase):
    def test_same_bounds(self):
        cs = config_space_nd(4, lower=-4, upper=5, log=False)
        hps = list(cs.values())

        for hp in hps:
            self.assertIsInstance(hp, CSH.NumericalHyperparameter)
            self.assertEqual(-4, hp.lower)
            self.assertEqual(5, hp.upper)
            self.assertFalse(hp.log)

    def test_prefix(self):
        # Default prefix
        cs = config_space_nd(4)
        hps = list(cs.values())

        expected_names = {"x1", "x2", "x3", "x4"}
        names = {hp.name for hp in hps}
        self.assertSetEqual(expected_names, names)

        # Other prefix
        cs = config_space_nd(4, variable_prefix="other_prefix_")
        hps = list(cs.values())

        expected_names = {"other_prefix_1", "other_prefix_2", "other_prefix_3", "other_prefix_4"}
        names = {hp.name for hp in hps}
        self.assertSetEqual(expected_names, names)

    def test_different_bounds(self):
        cs = config_space_nd(3, lower=(0, -1.5, -2), upper=(5, 20, 32.3))
        hps = list(cs.values())

        # Check Hyperparameter 0
        self.assertIsInstance(hps[0], CSH.NumericalHyperparameter)
        self.assertEqual(0, hps[0].lower)
        self.assertEqual(5, hps[0].upper)

        # Check Hyperparameter 1
        self.assertIsInstance(hps[1], CSH.NumericalHyperparameter)
        self.assertEqual(-1.5, hps[1].lower)
        self.assertEqual(20, hps[1].upper)

        # Check Hyperparameter 2
        self.assertIsInstance(hps[2], CSH.NumericalHyperparameter)
        self.assertEqual(-2, hps[2].lower)
        self.assertEqual(32.3, hps[2].upper)

    def test_constants(self):
        cs = config_space_nd(3, lower=(0, 5, -2.32), upper=(0, 5, -2.32))
        hps = list(cs.values())

        # Check Hyperparameter 0
        self.assertIsInstance(hps[0], CSH.Constant)
        self.assertEqual(0, hps[0].value)

        # Check Hyperparameter 1
        self.assertIsInstance(hps[1], CSH.Constant)
        self.assertEqual(5, hps[1].value)

        # Check Hyperparameter 2
        self.assertIsInstance(hps[2], CSH.Constant)
        self.assertEqual(-2.32, hps[2].value)


class TestLevy(TestCase):
    def test_config_space(self):
        f = Levy()
        default_cs = f.config_space
        hp = default_cs["x1"]
        self.assertIsInstance(hp, CSH.NumericalHyperparameter)
        self.assertEqual(-10, hp.lower)
        self.assertEqual(10, hp.upper)
        self.assertFalse(hp.log)

    def test_levy1D(self):
        f = Levy.for_n_dimensions(1)

        self.assertAlmostEqual(f(x1=1), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            y = f(x1=(np.random.random(1)[0] - 0.5) * 10 * 2)
            self.assertGreater(y, 0)

    def test_levy2D(self):
        f = Levy.for_n_dimensions(2, lower=-10, upper=10)
        self.assertAlmostEqual(f(x1=1, x2=1), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            x1, x2 = (np.random.random(2) - 0.5) * 10 * 2
            y = f(x1=x1, x2=x2)
            self.assertGreater(y, 0)


class TestAckley(TestCase):
    def test_config_space(self):
        f = Ackley()
        default_cs = f.config_space
        hp = default_cs["x1"]
        self.assertIsInstance(hp, CSH.NumericalHyperparameter)
        self.assertEqual(-32.768, hp.lower)
        self.assertEqual(32.768, hp.upper)
        self.assertFalse(hp.log)

    def test_ackley1D(self):
        f = Ackley.for_n_dimensions(1)

        self.assertAlmostEqual(f(x1=0), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            y = f(x1=(np.random.random(1)[0] - 0.5) * 32.768 * 2)
            self.assertGreater(y, 0)

    def test_ackley2D(self):
        f = Ackley.for_n_dimensions(2)
        self.assertAlmostEqual(f(x1=0, x2=0), 0)  # Minimum

        # Cannot be smaller than 0
        for _ in range(10000):
            x1, x2 = (np.random.random(2) - 0.5) * 10 * 2
            y = f(x1=x1, x2=x2)
            self.assertGreater(y, 0)


class TestCrossInTray(TestCase):
    def test_config_space(self):
        f = CrossInTray()
        default_cs = f.config_space
        hp = default_cs["x1"]
        self.assertIsInstance(hp, CSH.NumericalHyperparameter)
        self.assertEqual(-10, hp.lower)
        self.assertEqual(10, hp.upper)
        self.assertFalse(hp.log)

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

    def test_config_space(self):
        f = StyblinskiTang()
        default_cs = f.config_space
        hp = default_cs["x1"]
        self.assertIsInstance(hp, CSH.NumericalHyperparameter)
        self.assertEqual(-5, hp.lower)
        self.assertEqual(5, hp.upper)
        self.assertFalse(hp.log)

    def test_styblinski_tang_1D(self):
        f = StyblinskiTang.for_n_dimensions(1)
        self.assertAlmostEqual(f(x1=self.minimum_at), self.minimum)

    def test_minima(self):
        for d in range(1, 11):  # Test multiple dimensions
            f = StyblinskiTang.for_n_dimensions(d)
            x = {f"x{i + 1}": self.minimum_at for i in range(d)}
            print(f"Dimensions: {d:2d}, Input: {x}")
            self.assertEqual(len(x), d)
            self.assertAlmostEqual(f(**x), d * self.minimum)

    def test_simple_integral_numerical(self):
        f = StyblinskiTang()
        cs = f.config_space
        hp = cs["x1"]
        f_int = f.pd_integral(hp)
        integral_formula_1_value = f_int()
        integral_formula_2_value = StyblinskiTang._styblinski_tang_integral(
            hp.upper) - StyblinskiTang._styblinski_tang_integral(hp.lower)
        print(integral_formula_1_value)

        # Calculate ground truth by approximating it with sum of small rectangles
        n_steps = 5000
        integral_numeric = 0
        step_size = (hp.upper - hp.lower) / n_steps
        for i in range(n_steps):
            integral_numeric += f(x1=hp.lower + (i + 0.5) * step_size)

        integral_numeric *= step_size

        print("Numeric:", integral_numeric)
        print("Partial Dependence Function:", integral_formula_1_value)
        print("1D-Integral Function:", integral_formula_2_value)

        self.assertAlmostEqual(integral_numeric, integral_formula_1_value * (hp.upper - hp.lower), places=3)
        self.assertAlmostEqual(integral_numeric, integral_formula_2_value, places=3)

    def test_integral_1d(self):
        """
        f(x1, x2, x3) = stybli..
        F(x1) = f(x1, x2, x3) dx2 dx3
        """

        def styblinski_tang_3D_int_1D(x1: float, lower_x2: float = -5, upper_x2: float = 5, lower_x3: float = -5,
                                      upper_x3: float = 5) -> float:
            styblinski_tang = StyblinskiTang.for_n_dimensions(1)
            term_x1_lower_lower = styblinski_tang(x1=x1) * lower_x2 * lower_x3
            term_x1_lower_upper = styblinski_tang(x1=x1) * lower_x2 * upper_x3
            term_x1_upper_lower = styblinski_tang(x1=x1) * upper_x2 * lower_x3
            term_x1_upper_upper = styblinski_tang(x1=x1) * upper_x2 * upper_x3
            term_x1 = term_x1_upper_upper - term_x1_upper_lower - term_x1_lower_upper + term_x1_lower_lower

            styblinski_tang_integral = StyblinskiTang._styblinski_tang_integral

            term_x2_lower_lower = styblinski_tang_integral(lower_x2) * lower_x3
            term_x2_lower_upper = styblinski_tang_integral(lower_x2) * upper_x3
            term_x2_upper_lower = styblinski_tang_integral(upper_x2) * lower_x3
            term_x2_upper_upper = styblinski_tang_integral(upper_x2) * upper_x3
            term_x2 = term_x2_upper_upper - term_x2_upper_lower - term_x2_lower_upper + term_x2_lower_lower

            term_x3_lower_lower = styblinski_tang_integral(lower_x3) * lower_x2
            term_x3_lower_upper = styblinski_tang_integral(lower_x3) * upper_x2
            term_x3_upper_lower = styblinski_tang_integral(upper_x3) * lower_x2
            term_x3_upper_upper = styblinski_tang_integral(upper_x3) * upper_x2
            term_x3 = term_x3_upper_upper - term_x3_upper_lower - term_x3_lower_upper + term_x3_lower_lower

            return (term_x1 + term_x2 + term_x3) / ((upper_x2 - lower_x2) * (upper_x3 - lower_x3))

        f = StyblinskiTang.for_n_dimensions(3)
        f_int_specific = styblinski_tang_3D_int_1D
        f_int_general = f.pd_integral('x2', 'x3')
        x = np.linspace(-5, 5, num=100)

        for x1 in x:
            f_int_specific_x = f_int_specific(x1=x1)
            f_int_general_x = f_int_general(x1=x1)
            self.assertAlmostEqual(f_int_specific_x, f_int_general_x, places=5)

    def test_integral_2d_x3(self):
        """
        f(x1, x2, x3) = stybl...
        F(x1, x2) = f(x1, x2, x3) dx3
        """
        f = StyblinskiTang.for_n_dimensions(3)

        # Shortcuts
        def styblinski_tang_3D_int_2D(x1: float, x2: float, lower: float = -5, upper: float = 5) -> float:
            styblinski_tang_2D = StyblinskiTang.for_n_dimensions(2)
            lower_term = styblinski_tang_2D(x1=x1, x2=x2) * lower + f._styblinski_tang_integral(lower)
            upper_term = styblinski_tang_2D(x1=x1, x2=x2) * upper + f._styblinski_tang_integral(upper)
            return (upper_term - lower_term) / (upper - lower)  # normalization

        f_int_specific = styblinski_tang_3D_int_2D
        f_int_general = f.pd_integral(f.config_space["x3"])

        for x1 in np.linspace(-5, 5, num=100):
            for x2 in np.linspace(-5, 5, num=100):
                f_int_specific_x = f_int_specific(x1=x1, x2=x2)
                f_int_general_x = f_int_general(x1=x1, x2=x2)
                self.assertAlmostEqual(f_int_specific_x, f_int_general_x, places=5)

    def test_integral_2d_x2(self):
        """
        f(x1, x2, x3) = stybl...
        F(x1, x3) = f(x1, x2, x3) dx2
        """
        f = StyblinskiTang.for_n_dimensions(3)

        # Shortcuts
        def styblinski_tang_3D_int_2D(x1: float, x3: float, lower: float = -5, upper: float = 5) -> float:
            styblinski_tang_2D = StyblinskiTang.for_n_dimensions(2)
            lower_term = styblinski_tang_2D(x1=x1, x2=x3) * lower + f._styblinski_tang_integral(lower)
            upper_term = styblinski_tang_2D(x1=x1, x2=x3) * upper + f._styblinski_tang_integral(upper)
            return (upper_term - lower_term) / (upper - lower)  # normalization

        f_int_specific = styblinski_tang_3D_int_2D
        f_int_general = f.pd_integral(f.config_space["x2"])

        for x1 in np.linspace(-5, 5, num=100):
            for x3 in np.linspace(-5, 5, num=100):
                f_int_specific_x = f_int_specific(x1=x1, x3=x3)
                f_int_general_x = f_int_general(x1=x1, x3=x3)
                self.assertAlmostEqual(f_int_specific_x, f_int_general_x, places=5)


class TestPlotBlackboxFunctions(PlottableTest):
    def _apply_blackbox_plot(self, f: callable, cs: CS.ConfigurationSpace, name: str, **kwargs):
        if self.fig is None:
            self.initialize_figure()
        plot_function(f, cs, **kwargs)

        plt.title(name)
        plt.tight_layout()

    def test_plot_ackley_1D_zoomed(self):
        f = Ackley.for_n_dimensions(1, lower=-10, upper=10)
        cs = f.config_space
        self._apply_blackbox_plot(f, cs, "Ackley 1D")

    def test_plot_ackley_2D_zoomed(self):
        f = Ackley.for_n_dimensions(2, lower=-10, upper=10)
        cs = f.config_space
        self._apply_blackbox_plot(f, cs, "Ackley 1D")

    def test_plot_styblinski_tang_3D_int_1D(self):
        f = StyblinskiTang.for_n_dimensions(3)
        f_int = f.pd_integral('x2', 'x3')
        self._apply_blackbox_plot(f_int, f_int.config_space, "Styblinski Tang Integral 1D")

    def test_plot_styblinski_tang_3D_int_2D(self):
        f = StyblinskiTang.for_n_dimensions(3)
        f_int = f.pd_integral('x3')
        self._apply_blackbox_plot(f_int, f_int.config_space, "Styblinski Tang Integral 2D")

    def test_plot_styblinski_tang_integral(self):
        f = StyblinskiTang.for_n_dimensions(2)
        f_int = f.pd_integral('x2')
        self._apply_blackbox_plot(f_int, f_int.config_space, "Styblinski Tang Integral 1D")

    def test_integral_function(self):
        cs = CS.ConfigurationSpace()  # Cannot use config_space_nd, because function takes "x" instead of "x1" as input
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter("x", -5, 5))
        self._apply_blackbox_plot(StyblinskiTang._styblinski_tang_integral, cs, "Styblinski Tang Integral Function 1D")


@pytest.mark.parametrize("f", [
    Square.for_n_dimensions(1), Square.for_n_dimensions(2),
    NegativeSquare.for_n_dimensions(1), NegativeSquare.for_n_dimensions(2),
    Ackley.for_n_dimensions(1), Ackley.for_n_dimensions(2),
    CrossInTray(),
    Levy.for_n_dimensions(1), Levy.for_n_dimensions(2),
    StyblinskiTang.for_n_dimensions(1), StyblinskiTang.for_n_dimensions(2)
])
def test_plot_all(f: BlackboxFunction):
    plt.figure(figsize=(16, 9))
    cs = f.config_space
    plot_function(f, cs)

    plt.title(str(f))
    plt.tight_layout()
    plt.savefig(TestPlotBlackboxFunctions.SAVE_FOLDER / TestPlotBlackboxFunctions.__name__ / f"{str(f)}.png")
    plt.show()
