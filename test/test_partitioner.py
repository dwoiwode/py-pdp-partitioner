import unittest

from matplotlib import pyplot as plt

from src.blackbox_functions import square_2D
from src.config_spaces import square_2D_config_space
from src.optimizer import BayesianOptimization
from src.partitioner import DecisionTreePartitioner
from src.pdp import PDP
from src.plotting import plot_ice


class TestPartitioner(unittest.TestCase):
    def test_dt_partitioner_2d(self):
        bo = BayesianOptimization(square_2D, config_space=square_2D_config_space())
        # partitioner = DecisionTreePartitioner()

        bo.optimize(10)
        pdp = PDP(bo)
        idx = 0
        x_ice, y_ice, variances = pdp.calculate_ice(idx)

        partitioner = DecisionTreePartitioner(idx, x_ice, variances)
        partitioner.partition()

        left_indices = partitioner.root.left_child.index_arr
        right_indices = partitioner.root.right_child.index_arr

        ax = plot_ice(x_ice[left_indices], y_ice[left_indices], idx)
        plot_ice(x_ice[right_indices], y_ice[right_indices], idx, ax=ax, color='blue')
        plt.show()
