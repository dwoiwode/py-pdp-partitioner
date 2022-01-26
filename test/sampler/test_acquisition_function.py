import matplotlib.pyplot as plt
import numpy as np

from src.blackbox_functions import config_space_nd
from src.sampler.acquisition_function import ExpectedImprovement, ProbabilityOfImprovement, LowerConfidenceBound
from src.surrogate_models import GaussianProcessSurrogate
from test import PlottableTest


class TestAcquisitionFunctions(PlottableTest):
    def setUp(self) -> None:
        super().setUp()
        self.cs = config_space_nd(1)
        self.surrogate = GaussianProcessSurrogate(self.cs)
        self.X = np.asarray([-2, -1, 1, 2, 3, 4]).reshape((-1, 1))
        self.y = np.asarray([4, 1, 1, 4, 9, 16])
        self.X_scaled = (self.X + 5) / 10

    def test_expected_improvement(self):
        self.initialize_figure()

        ei = ExpectedImprovement(self.cs, self.surrogate, samples_for_optimization=2000)

        self.surrogate.fit(self.X_scaled, self.y)
        ei.update(1)

        # Plotting
        fig = self.fig
        assert isinstance(fig, plt.Figure)
        ax_function = fig.add_subplot(3, 1, (1, 2))
        ax_acquisition = fig.add_subplot(3, 1, 3)
        assert isinstance(ax_function, plt.Axes)
        assert isinstance(ax_acquisition, plt.Axes)

        self.surrogate.plot(ax=ax_function)
        ax_function.plot(self.X, self.y, ".", color="red")  # Plot samples
        ei.plot(ax=ax_acquisition)
        self.save_fig()

        self.assertAlmostEqual(0, ei.get_optimum()["x1"], places=1)

    def test_probability_of_improvement(self):
        self.initialize_figure()
        pi = ProbabilityOfImprovement(self.cs, self.surrogate, eps=1, samples_for_optimization=2000)

        self.surrogate.fit(self.X_scaled, self.y)
        pi.update(1)

        # Plotting
        fig = self.fig
        assert isinstance(fig, plt.Figure)
        ax_function = fig.add_subplot(3, 1, (1, 2))
        ax_acquisition = fig.add_subplot(3, 1, 3)
        assert isinstance(ax_function, plt.Axes)
        assert isinstance(ax_acquisition, plt.Axes)

        self.surrogate.plot(ax=ax_function)
        ax_function.plot(self.X, self.y, ".", color="red")  # Plot samples
        pi.plot(ax=ax_acquisition)
        self.save_fig()

        self.assertAlmostEqual(0, pi.get_optimum()["x1"], places=1)

    def test_lower_confidence_bound(self):
        self.initialize_figure()

        lcb = LowerConfidenceBound(self.cs, self.surrogate, samples_for_optimization=2000)

        self.surrogate.fit(self.X_scaled, self.y)
        lcb.update(1)

        # Plotting
        fig = self.fig
        assert isinstance(fig, plt.Figure)
        ax_function = fig.add_subplot(3, 1, (1, 2))
        ax_acquisition = fig.add_subplot(3, 1, 3)
        assert isinstance(ax_function, plt.Axes)
        assert isinstance(ax_acquisition, plt.Axes)

        self.surrogate.plot(ax=ax_function)
        ax_function.plot(self.X, self.y, ".", color="red")  # Plot samples
        lcb.plot(ax=ax_acquisition)
        self.save_fig()

        self.assertAlmostEqual(-5, lcb.get_optimum()["x1"], places=1)
