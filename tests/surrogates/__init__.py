import abc
import unittest
from typing import Type

from blackbox_functions.synthetic_functions import StyblinskiTang
from sampler.random_sampler import RandomSampler
from surrogate_models import SurrogateModel


class SurrogateTest(unittest.TestCase, abc.ABC):
    surrogate_class: Type[SurrogateModel]

    def setUp(self) -> None:
        # Blackbox function
        f = StyblinskiTang.for_n_dimensions(3)
        self.cs = f.config_space

        # Sampling
        sampler = RandomSampler(f, self.cs)
        sampler.sample(100)

        # Surrogate
        self.surrogate: SurrogateModel = self.surrogate_class(self.cs)
        self.surrogate.fit(sampler.X, sampler.y)

    def test_predict_config(self):
        # Setup
        config = self.cs.sample_configuration()
        mean, std = self.surrogate.predict_config(config)

        # Tests
        self.assertIsInstance(mean, float)
        self.assertIsInstance(std, float)

    def test_predict_configs(self):
        # Setup
        configs = self.cs.sample_configuration(100)
        means, stds = self.surrogate.predict_configs(configs)

        # Tests
        self.assertIsInstance(means, list)
        self.assertIsInstance(stds, list)
        self.assertEqual(100, len(means))
        self.assertEqual(100, len(stds))

        for mean, std in zip(means, stds):
            self.assertIsInstance(mean, float)
            self.assertIsInstance(std, float)

