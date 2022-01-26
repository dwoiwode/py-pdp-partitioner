from unittest import TestCase

from ConfigSpace import Configuration

from src.algorithms.partitioner.decision_tree_partitioner import SplitCondition
from src.blackbox_functions import config_space_nd


class TestSplitCondition(TestCase):
    def test_float_value(self):
        cs = config_space_nd(1, upper=5, lower=-5)
        hyperparameter = cs.get_hyperparameter('x1')
        value = 3
        cond = SplitCondition(cs, hyperparameter, value=value, less_equal=True)

        self.assertEqual(cond.normalized_value, 0.8)
        self.assertEqual(cond.value, 3)
        self.assertEqual(str(cond), 'x1 <= 3.0')

        test_config = Configuration(cs, values={'x1': -1})
        self.assertEqual(test_config.get('x1'), -1)
        self.assertTrue(cond.is_satisfied(test_config))

        test_config = Configuration(cs, values={'x1': 3})
        self.assertTrue(cond.is_satisfied(test_config))

        test_config = Configuration(cs, values={'x1': 5})
        self.assertFalse(cond.is_satisfied(test_config))

    def test_normalized_value(self):
        cs = config_space_nd(1, upper=5, lower=-5)
        hyperparameter = cs.get_hyperparameter('x1')
        normalized_value = 0.4
        cond = SplitCondition(cs, hyperparameter, normalized_value=normalized_value, less_equal=False)

        self.assertEqual(cond.normalized_value, 0.4)
        self.assertEqual(cond.value, -1)
        self.assertEqual(str(cond), 'x1 > -1.0')

        test_config = Configuration(cs, values={'x1': 3})
        self.assertTrue(cond.is_satisfied(test_config))

        test_config = Configuration(cs, values={'x1': 0})
        self.assertTrue(cond.is_satisfied(test_config))

        test_config = Configuration(cs, values={'x1': -1})
        self.assertFalse(cond.is_satisfied(test_config))

        test_config = Configuration(cs, values={'x1': -5})
        self.assertFalse(cond.is_satisfied(test_config))





