import matplotlib.pyplot as plt
import numpy as np

from src.demo_data.config_spaces import config_space_nd
from src.sampler.acquisition_function import ExpectedImprovement, ProbabilityOfImprovement, LowerConfidenceBound
from src.surrogate_models import GaussianProcessSurrogate
from test import PlottableTest


class TestAcquisitionFunctions(PlottableTest):
    def test_expected_improvement(self):
        self.initialize_figure()
        cs = config_space_nd(1)
        selected_hyperparameter = cs.get_hyperparameter("x1")

        surrogate = GaussianProcessSurrogate(cs)
        ei = ExpectedImprovement(cs, surrogate, samples_for_optimization=2000)

        X = np.asarray([-2, -1, 1, 2, 3, 4]).reshape((-1, 1))
        y = [4, 1, 1, 4, 9, 16]
        X_scaled = (X + 5) / 10
        surrogate.fit(X_scaled, y)
        ei.update(1)

        # Plotting
        fig = self.fig
        assert isinstance(fig, plt.Figure)
        ax_function = fig.add_subplot(3, 1, (1, 2))
        ax_acquisition = fig.add_subplot(3, 1, 3)
        assert isinstance(ax_function, plt.Axes)
        assert isinstance(ax_acquisition, plt.Axes)

        surrogate.plot(ax=ax_function, x_hyperparameters=selected_hyperparameter)
        ax_function.plot(X, y, ".", color="red")  # Plot samples
        ei.plot(ax=ax_acquisition)

        self.assertAlmostEqual(0, ei.get_optimum()["x1"], places=1)

    def test_probability_of_improvement(self):
        self.initialize_figure()
        cs = config_space_nd(1)
        selected_hyperparameter = cs.get_hyperparameter("x1")

        surrogate = GaussianProcessSurrogate(cs)
        pi = ProbabilityOfImprovement(cs, surrogate, samples_for_optimization=2000)

        X = np.asarray([-2, -1, 1, 2, 3, 4]).reshape((-1, 1))
        y = [4, 1, 1, 4, 9, 16]
        X_scaled = (X + 5) / 10
        surrogate.fit(X_scaled, y)
        pi.update(1)

        # Plotting
        fig = self.fig
        assert isinstance(fig, plt.Figure)
        ax_function = fig.add_subplot(3, 1, (1, 2))
        ax_acquisition = fig.add_subplot(3, 1, 3)
        assert isinstance(ax_function, plt.Axes)
        assert isinstance(ax_acquisition, plt.Axes)

        surrogate.plot(ax=ax_function, x_hyperparameters=selected_hyperparameter)
        ax_function.plot(X, y, ".", color="red")  # Plot samples
        pi.plot(ax=ax_acquisition)

        self.assertAlmostEqual(0, pi.get_optimum()["x1"], places=1)

    def test_lower_confidence_bound(self):
        self.initialize_figure()
        cs = config_space_nd(1)
        selected_hyperparameter = cs.get_hyperparameter("x1")

        surrogate = GaussianProcessSurrogate(cs)
        lcb = LowerConfidenceBound(cs, surrogate, samples_for_optimization=2000)

        X = np.asarray([-2, -1, 1, 2, 3, 4]).reshape((-1, 1))
        y = [4, 1, 1, 4, 9, 16]
        X_scaled = (X + 5) / 10
        surrogate.fit(X_scaled, y)
        lcb.update(1)

        # Plotting
        fig = self.fig
        assert isinstance(fig, plt.Figure)
        ax_function = fig.add_subplot(3, 1, (1, 2))
        ax_acquisition = fig.add_subplot(3, 1, 3)
        assert isinstance(ax_function, plt.Axes)
        assert isinstance(ax_acquisition, plt.Axes)

        surrogate.plot(ax=ax_function, x_hyperparameters=selected_hyperparameter)
        ax_function.plot(X, y, ".", color="red")  # Plot samples
        lcb.plot(ax=ax_acquisition)

        self.assertAlmostEqual(-5, lcb.get_optimum()["x1"], places=1)
