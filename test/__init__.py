from pathlib import Path
from typing import Optional, Tuple
from unittest import TestCase

from matplotlib import pyplot as plt


class PlottableTest(TestCase):
    SHOW = True
    SAVE_FOLDER = Path(__file__).parent / "plots"

    def setUp(self) -> None:
        # Make sure that figure is cleared from previous tests
        plt.clf()
        self.fig: Optional[plt.Figure] = None
        self.fig_idx = 0
        self._last_save_fig = None

    def tearDown(self) -> None:
        # Save figure from last test
        if self.fig is None:
            return

        if self.SAVE_FOLDER is not None:
            self.save_fig(iterate=False)

        # Show plot from last test
        if self.SHOW:
            plt.show()
        plt.close()

    def save_fig(self, iterate=True, name=None):
        if self.fig is None:
            raise ValueError("Figure is None")
        if self._last_save_fig == self.fig:
            return  # Already saved this

        if name is None:
            name = self._testMethodName

        # Create folder
        folder = self.SAVE_FOLDER / self.__class__.__name__
        folder.mkdir(parents=True, exist_ok=True)

        # Save fig
        self.fig.legend()  # Add legend
        plt.tight_layout()
        if self.fig_idx == 0 and not iterate:
            self.fig.savefig(folder / f"{name}.png")
        else:
            self.fig_idx += 1
            self.fig.savefig(folder / f"{name}_{self.fig_idx}.png")

    def initialize_figure(self, figsize: Tuple[int, int] = (16, 9)):
        plt.clf()
        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle(self._testMethodName, fontsize=16)
