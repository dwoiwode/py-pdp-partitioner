from unittest import TestCase
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration

from src.utils.utils import scale_float, unscale_float


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
