import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Tuple, Optional, Union

import ConfigSpace as CS
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm

from pyPDP.utils.plotting import get_ax, check_and_set_axis
from pyPDP.utils.typing import ColorType, SelectedHyperparameterType
from pyPDP.utils.utils import config_list_to_array, get_hyperparameters, median_distance_between_points, \
    ConfigSpaceHolder, copy_config_space, ProgressDummy


class Sampler(ConfigSpaceHolder, ABC):
    CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "sampler"

    def __init__(
            self,
            obj_func: Callable,
            config_space: CS.ConfigurationSpace,
            minimize_objective=True,
            seed=None
    ):
        super().__init__(config_space, seed=seed)
        self.obj_func = obj_func

        self.minimize_objective = minimize_objective
        self.hash = self._hash(seed)

        self.config_list: List[CS.Configuration] = []
        self.y_list: List[float] = []
        self._cache: List[Tuple[CS.Configuration, float]] = []
        self._load_cache()

    def __str__(self):
        return f"{self.__class__.__name__}({self.obj_func.__name__})"

    def __len__(self) -> int:
        return len(self.config_list)

    def __del__(self):
        self.save_cache()

    def _hash(self, *args) -> str:
        md = hashlib.md5()
        md.update(bytes(str(self.__class__), encoding="latin"))
        md.update(bytes(str(self.config_space.get_hyperparameters()), encoding="latin"))
        for arg in args:
            md.update(bytes(str(arg), encoding="latin"))
        md.update(bytes(str(self.obj_func), encoding="latin"))
        return md.hexdigest()

    def clear_cache(self):
        self._cache = []

    def reset(self):
        self.config_list = []
        self.y_list = []

    @property
    def X(self) -> np.ndarray:
        return config_list_to_array(self.config_list)

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

    @property
    def incumbent_config(self) -> Optional[CS.Configuration]:
        return self.incumbent[0]

    @property
    def incumbent_value(self) -> float:
        return self.incumbent[1]

    def _load_cache(self):
        self._cache = []
        file = self.CACHE_DIR / f"{self.hash}.json"
        if not file.exists():
            self.logger.debug("Cache does not exist")
            return

        try:
            self.logger.debug("Loading cache...")
            data = json.loads(file.read_text())
            for X, y in zip(data["X"], data["y"]):
                config = CS.Configuration(self.config_space, vector=X, origin="Cache")
                self._cache.append((config, y))
            self.logger.debug(f"Loaded {len(self._cache)} samples from cache")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.warning(f"Loading cache failed: {e}")

    def save_cache(self):
        if len(self._cache) > 0:
            # Current version of sampler is worse than cache
            return

        if len(self) == 0:
            # We did not sample anything
            return

        # Save in cache
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        file = self.CACHE_DIR / f"{self.hash}.json"
        self.logger.info(f"Writing cache to {file} ({len(self)} samples)")
        file.write_text(json.dumps(
            {
                "X": self.X.tolist(),
                "y": self.y_list
            },
            separators=(",", ":")
        ), encoding="latin")

    def sample(self, n_points: int = 1, *, show_progress=False):
        """ Samples n_points new points """
        # if more points than cached previously, resample
        if show_progress:
            progress_bar = tqdm(total=n_points, desc=f"{self}")
        else:
            progress_bar = ProgressDummy()  # Does nothing except existing

        if len(self) + n_points > len(self._cache):
            self._cache = []

        # Use cache
        sampled_points = 0
        while len(self._cache) > 0 and sampled_points < n_points:
            config, value = self._cache.pop(0)

            self.config_list.append(config)
            self.y_list.append(value)
            sampled_points += 1
            progress_bar.update(1)
            progress_bar.refresh()

        # Use sample function
        if sampled_points < n_points:
            self._sample(n_points - sampled_points, pbar=progress_bar)

        progress_bar.close()

    @abstractmethod
    def _sample(self, n_points: int = 1, pbar: Union[ProgressDummy, tqdm] = ProgressDummy()):
        """ Samples n_points new points """
        pass

    def maximum_mean_discrepancy(self, m: Optional[int] = None, seed=None) -> float:
        # Get and transform samples
        if m is None:
            m = len(self)
        X_samples = config_list_to_array(self.X)

        new_cs = copy_config_space(self.config_space, seed=seed)
        X_uniform = config_list_to_array(new_cs.sample_configuration(m))
        X = np.concatenate((X_samples, X_uniform))

        # Calculate
        # X_unscaled = unscale(X, self.config_space)
        median_l2 = median_distance_between_points(X)
        # median_l2 = median_distance_between_points(X_unscaled)
        rbf = RBF(median_l2)

        covariances = rbf(X)
        # covariances = rbf(X_unscaled)
        n = len(X_samples)
        term1 = np.sum((1 - np.eye(n)) * covariances[:n, :n]) / (n * (n - 1))
        term2 = np.sum((1 - np.eye(m)) * covariances[n:, n:]) / (m * (m - 1))
        term3 = np.sum(covariances[:n, n:]) * 2 / (n * m)

        term_sum = term1 + term2 - term3
        if term_sum < 0:
            return 0
        return np.sqrt(term_sum)

    def plot(self,
             color: ColorType = "red",
             marker: str = ".",
             label: Optional[str] = None,
             *,  # Prevent next args to be added via positional arguments
             x_hyperparameters: SelectedHyperparameterType = None,
             ax: Optional[plt.Axes] = None):
        # Resolve arguments
        ax = get_ax(ax)
        x_hyperparameters = get_hyperparameters(x_hyperparameters, self.config_space)
        if label is None:
            label = f"Sampled points ({len(self)}, {self.__class__.__name__})"

        # Check whether plot is possible
        check_and_set_axis(ax, x_hyperparameters)

        # Plot
        plotting_kwargs = {
            "marker": marker,
            "color": color,
            "label": label
        }

        n_hyperparameters = len(x_hyperparameters)
        if n_hyperparameters == 1:  # 1D
            plotting_kwargs["linestyle"] = ""

            hp = x_hyperparameters[0]
            x = np.asarray([config[hp.name] for config in self.config_list])
            order = np.argsort(x)
            ax.plot(x[order], self.y[order], **plotting_kwargs)
        elif n_hyperparameters == 2:  # 2D
            hp1, hp2 = x_hyperparameters
            x1, x2 = zip(*[(config[hp1.name], config[hp2.name]) for config in self.config_list])
            # colors = self.y  # TODO: How to plot values? color=colors is possible, not visible on ground truth
            ax.scatter(x1, x2, **plotting_kwargs)
        else:
            raise NotImplementedError(f"Plotting for {n_hyperparameters} dimensions not implemented. "
                                      "Please select a specific hp by setting `x_hyperparemeters`")
