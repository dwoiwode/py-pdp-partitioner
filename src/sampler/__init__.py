from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Optional, Iterable

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt

from src.utils.plotting import Plottable, get_ax, check_and_set_axis
from src.utils.typing import ColorType
from src.utils.utils import config_list_to_2d_arr, get_hyperparameters


class Sampler(Plottable, ABC):
    def __init__(self,
                 obj_func: Callable,
                 config_space: CS.ConfigurationSpace,
                 minimize_objective=True,
                 seed=None):
        super().__init__()
        self.obj_func = obj_func
        self.config_space = config_space
        self.minimize_objective = minimize_objective
        self.seed = seed

        self.config_list: List[CS.Configuration] = []
        self.y_list: List[float] = []

    def __len__(self) -> int:
        return len(self.config_list)

    def reset(self):
        self.config_list = []
        self.y_list = []

    @property
    def X(self) -> np.ndarray:
        return config_list_to_2d_arr(self.config_list)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self.y_list)

    @property
    def incumbent(self) -> Tuple[Optional[CS.Configuration], float]:
        if len(self.y_list) == 0:
            return None, float("inf")
        if self.minimize_objective:
            incumbent_index = np.argmin(self.y_list)
        else:
            incumbent_index = np.argmax(self.y_list)

        incumbent_config = self.config_list[incumbent_index]
        incumbent_value = self.y_list[incumbent_index]
        return incumbent_config, incumbent_value

    @abstractmethod
    def sample(self, n_points: int = 1):
        """ Samples n_points new points """
        pass

    def plot(self,
             color: ColorType = "red",
             marker: str = ".",
             label: Optional[str] = None,
             *,  # Prevent next args to be added via positional arguments
             x_hyperparameters: Optional[Iterable[CSH.Hyperparameter]] = None,
             ax: Optional[plt.Axes] = None):
        # Resolve arguments
        ax = get_ax(ax)
        x_hyperparameters = get_hyperparameters(x_hyperparameters, self.config_space)
        if label is None:
            label = f"Sampled points ({self.__class__.__name__})"

        # Check whether plot is possible
        check_and_set_axis(ax, x_hyperparameters)

        # Plot
        plotting_kwargs = {
            "marker": marker,
            "linestyle": "",
            "color": color,
            "label": label
        }

        n_hyperparameters = len(x_hyperparameters)
        if n_hyperparameters == 1:  # 1D
            hp = x_hyperparameters[0]
            x = np.asarray([config[hp.name] for config in self.config_list])
            order = np.argsort(x)
            ax.plot(x[order], self.y[order], **plotting_kwargs)
        elif n_hyperparameters == 2:  # 2D
            hp1, hp2 = x_hyperparameters
            x1, x2 = zip(*[(config[hp1.name], config[hp2.name]) for config in self.config_list])
            # colors = self.y  # TODO: How to plot values? color=colors is possible, not visible on ground truth
            ax.scatter(x1, x2)
        else:
            raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")
