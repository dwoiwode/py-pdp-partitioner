from typing import Optional, Callable, Any, Type
import ConfigSpace as CS

from src.optimizer import AbstractOptimizer, RandomSearch
from src.partitioner import AbstractPartitioner, DecisionTreePartitioner

import numpy as np

class PDP:
    def __init__(self,
                 partitioner: Optional[AbstractPartitioner],
                 optimizer: Optional[AbstractOptimizer]):
        self.partitioner = partitioner
        self.optimizer = optimizer

    def create(self):
        pass

    def calculate_ice(self, idx: int):
        X = np.asarray([config.get_array() for config in self.optimizer.config_list])
        num_instances, num_features = X.shape
        x_s = X[:, idx]
        X_ice = X.repeat(num_instances).reshape((num_instances, num_features, -1)).transpose((2, 0, 1))
        for i in range(num_instances):
            X_ice[i, :, idx] = x_s[i]
        means, stds = self.optimizer.surrogate_score(X_ice.reshape(-1, num_features))
        y_ice = means.reshape((num_instances, num_instances))
        return X_ice, y_ice

    # def prepare_ice(self, model, X, s, centered=False):
    #     """
    #     Uses `calculate_ice` to retrieve plot data.
    #
    #     Parameters:
    #         model: Classifier which can call a predict method.
    #         X (np.array with shape (num_instances, num_features)): Input data.
    #         s (int): Index of the feature x_s.
    #         centered (bool): Whether c-ICE should be used or not.
    #
    #     Returns:
    #         all_x (list or 1D np.ndarray): List of lists of the x values.
    #         all_y (list or 1D np.ndarray): List of lists of the y values.
    #             Each entry in `all_x` and `all_y` represents one line in the plot.
    #     """
    #     num_instances, num_features = X.shape
    #     all_x, all_y = self.calculate_ice(model, X, s)
    #     x_s = X[:, s]
    #     order = np.argsort(x_s)
    #
    #     all_x = x_s[order].repeat(num_instances).reshape((num_instances, num_instances))
    #     all_y = all_y[order].T
    #     all_x = all_x.T
    #
    #     if centered:
    #         all_y -= np.reshape(all_y[:, 0], (-1, 1))
    #
    #     return all_x, all_y
    #
    # def plot_ice(self, model, dataset, X, s, centered=False):
    #     """
    #     Creates a plot object and fills it with the content of `prepare_ice`.
    #     Note: `show` method is not called.
    #
    #     Parameters:
    #         model: Classifier which can call a predict method.
    #         dataset (utils.Dataset): Used dataset to train the model. Used to receive the labels.
    #         s (int): Index of the feature x_s.
    #         centered (bool): Whether c-ICE should be used or not.
    #
    #     Returns:
    #         plt (matplotlib.pyplot or utils.styled_plot.plt)
    #     """
    #
    #     plt.figure()
    #     all_x, all_y = prepare_ice(model, X, s, centered=centered)
    #     for x_values, y_values in zip(all_x, all_y):
    #         plt.plot(x_values, y_values, alpha=0.2)
    #
    #     plt.xlabel(dataset.get_input_labels(s))
    #     plt.ylabel(dataset.get_output_label())
    #
    #     return plt
    #
    # def prepare_pdp(self, model, X, s):
    #     """
    #     Uses `calculate_ice` to retrieve plot data for PDP.
    #
    #     Parameters:
    #         model: Classifier which can call a predict method.
    #         X (np.ndarray with shape (num_instances, num_features)): Input data.
    #         s (int): Index of the feature x_s.
    #
    #     Returns:
    #         x (list or 1D np.ndarray): x values of the PDP line.
    #         y (list or 1D np.ndarray): y values of the PDP line.
    #     """
    #     X_ice, y_ice = calculate_ice(model, X, s)
    #     x_s = X[:, s]
    #     order = np.argsort(x_s)
    #     x_s = x_s[order]
    #
    #     y = np.mean(y_ice[order].T, axis=0)
    #
    #     return x_s, y
