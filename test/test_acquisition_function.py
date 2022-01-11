import unittest

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from src.optimizer import ProbabilityOfImprovement, ExpectedImprovement
from src.plotting import plot_model_confidence, plot_samples, plot_acquisition
from test.test_plotting import TestPlotting


class TestAcquisitionFunctions(TestPlotting):
    @unittest.SkipTest
    def test_probability_of_improvement(self):
        pass

    def test_expected_improvement(self):
        x = CSH.UniformFloatHyperparameter("x", lower=-4, upper=4)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(x)

        gpr = GaussianProcessRegressor()
        ei = ExpectedImprovement(cs, gpr, samples=2000)

        X = np.asarray([-2, -1, 1, 2, 3, 4]).reshape((-1, 1))
        y = [4, 1, 1, 4, 9, 16]
        X_scaled = (X + 4) / 8
        print(X_scaled)
        gpr.fit(X_scaled, y)
        ei.update(1)

        def surrogate_score(x1):
            x1 = np.expand_dims(x1, axis=1)
            return gpr.predict(x1, return_std=True)

        # Plotting
        fig = plt.figure(figsize=(16, 9))
        assert isinstance(fig, plt.Figure)
        ax_function = fig.add_subplot(3, 1, (1, 2))
        ax_acquisition = fig.add_subplot(3, 1, 3)
        assert isinstance(ax_function, plt.Axes)
        assert isinstance(ax_acquisition, plt.Axes)

        plot_model_confidence(surrogate_score, cs, ax=ax_function)
        plot_samples([CS.Configuration(cs, values={"x": float(a)}) for a in X], y, ax=ax_function)
        plot_acquisition(ei, cs, ax=ax_acquisition)

        self.assertAlmostEqual(0, ei.get_optimum()["x"], places=2)


# def _probability_of_improvement(config: CS.Configuration, model, eta, exploration=0,
#                                 minimize_objective=True) -> CS.Configuration:
#     # sample points
#     x_sample_array = [config.get_array()]
#     means, stds = model.predict(x_sample_array, return_std=True)
#
#     # prob of improvement for sampled points
#     if minimize_objective:
#         temp = (eta - means - exploration) / stds
#     else:
#         temp = (means - eta - exploration) / stds
#     prob_of_improvement = norm.cdf(temp)
#
#     # best sampled point
#     return prob_of_improvement[0]
