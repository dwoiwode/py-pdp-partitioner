from unittest import TestCase
import ConfigSpace.hyperparameters as CSH
import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration

from pyPDP.utils.utils import scale_float, unscale_float, unscale, copy_config_space


class TestUtils(TestCase):
    def test_scale_float(self):
        hp = CSH.UniformFloatHyperparameter('x1', lower=-5, upper=5, log=False)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(hp)

        # scale manually
        x_unscaled = np.linspace(-5, 5, 1000)
        x_scaled = np.asarray([scale_float(x, cs, hp) for x in x_unscaled])

        # scale using configspace internals
        x_configs = [Configuration(cs, values={'x1': x}) for x in x_unscaled]
        x_scaled_config = np.asarray([config.get_array().item() for config in x_configs])

        self.assertTrue(np.array_equal(x_scaled, x_scaled_config))

    def test_scale_float_log(self):
        hp = CSH.UniformFloatHyperparameter('x1', lower=1, upper=100, log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(hp)

        # scale manually
        x_unscaled = np.linspace(1, 100, 1000)
        x_scaled = np.asarray([scale_float(x, cs, hp) for x in x_unscaled])

        # scale using configspace internals
        x_configs = [Configuration(cs, values={'x1': x}) for x in x_unscaled]
        x_scaled_config = np.asarray([config.get_array().item() for config in x_configs])

        self.assertTrue(np.array_equal(x_scaled, x_scaled_config))

    def test_unscale_float(self):
        hp = CSH.UniformFloatHyperparameter('x1', lower=-5, upper=5, log=False)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(hp)

        # scale manually
        x_unscaled = np.linspace(0, 1, 1000)
        x_scaled = np.asarray([unscale_float(x, cs, hp) for x in x_unscaled])

        # scale using configspace internals
        x_configs = [Configuration(cs, vector=np.asarray([x])) for x in x_unscaled]
        x_scaled_config = np.asarray([config.get('x1') for config in x_configs])

        self.assertTrue(np.array_equal(x_scaled, x_scaled_config))

    def test_unscale_float_log(self):
        hp = CSH.UniformFloatHyperparameter('x1', lower=1, upper=100, log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(hp)

        # scale manually
        x_unscaled = np.linspace(0, 1, 1000)
        x_scaled = np.asarray([unscale_float(x, cs, hp) for x in x_unscaled])

        # scale using configspace internals
        x_configs = [Configuration(cs, vector=np.asarray([x])) for x in x_unscaled]
        x_scaled_config = np.asarray([config.get('x1') for config in x_configs])

        self.assertTrue(np.array_equal(x_scaled, x_scaled_config))

    def test_unscale_categorical(self):
        hp = CSH.CategoricalHyperparameter('x', choices=["A", "B", "C", "D"])
        cs = ConfigurationSpace()
        cs.add_hyperparameter(hp)

        # scale manually
        x_unscaled = [[0], [2], [1], [3]]
        x_scaled = unscale(np.asarray(x_unscaled), cs)

        # scale using configspace internals
        x_configs = [Configuration(cs, vector=np.asarray([x])) for x in x_unscaled]
        x_scaled_config = np.asarray([[config.get('x')] for config in x_configs])

        self.assertTrue(np.array_equal(x_scaled, x_scaled_config))

    def test_copy_cs_categorical(self):
        hp = CSH.CategoricalHyperparameter('x', choices=["A", "B", "C", "D"])
        hp2 = CSH.UniformFloatHyperparameter('y', lower=-0.7352, upper=32.32)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(hp)
        cs.add_hyperparameter(hp2)

        cs2 = copy_config_space(cs)
        self.assertEqual(cs, cs2)
        self.assertFalse(cs is cs2)
