import hashlib
import warnings
from typing import Callable, Any, List, Tuple, Union, Optional

import ConfigSpace as CS
import numpy as np
from tqdm import tqdm

from pyPDP.sampler import Sampler
from pyPDP.sampler.acquisition_function import LowerConfidenceBound, AcquisitionFunction
from pyPDP.surrogate_models import SurrogateModel
from pyPDP.surrogate_models.sklearn_surrogates import GaussianProcessSurrogate
from pyPDP.utils.utils import ProgressDummy


class BayesianOptimizationSampler(Sampler):
    def __init__(self,
                 obj_func: Callable[[Any], float],
                 config_space: CS.ConfigurationSpace,
                 surrogate_model: Optional[SurrogateModel] = None,
                 initial_points: int = 5,
                 acq_class=None,
                 acq_class_kwargs=None,
                 minimize_objective: bool = True,
                 seed=None):
        super().__init__(obj_func, config_space, minimize_objective, seed=seed)
        # Initialize class
        self.initial_points = initial_points  # number of initial points to be sampled

        # Surrogate model
        if surrogate_model is None:
            surrogate_model = GaussianProcessSurrogate(self.config_space, seed=seed)
            self.logger.info(f"Surrogate model is None. Taking {surrogate_model} as default")
        self.surrogate_model = surrogate_model
        self._model_fitted_hash: str = ""

        # Acquisition function
        if acq_class_kwargs is None:
            acq_class_kwargs = {}
        if acq_class is None:
            acq_class = LowerConfidenceBound  # Default Lower Confidence Bound
        self.acq_func: AcquisitionFunction = acq_class(self.config_space,
                                                       self.surrogate_model,
                                                       minimize_objective=minimize_objective,
                                                       seed=seed,
                                                       **acq_class_kwargs)

        # Update cache according to additional arguments
        self.hash = self._hash(seed, acq_class, acq_class_kwargs, initial_points, surrogate_model.__class__)
        self._load_cache()

    def _sample_initial_points(self, max_sampled_points=None):
        if max_sampled_points is None:
            sampled_points = self.initial_points
        else:
            sampled_points = min(self.initial_points, max_sampled_points)

        self.config_list = self.config_space.sample_configuration(sampled_points)
        if self.initial_points == 1:  # for a single value, the sampling does not return a list
            self.config_list = [self.config_list]

        self.y_list = [self.obj_func(**config) for config in self.config_list]
        self.fit_surrogate(force=True)

    def fit_surrogate(self, force: bool = False):
        """
        Fits the surrogate model. If force is False and surrogate model already fitted with current configs, do nothing
        """
        parameter_hash = hashlib.md5()
        parameter_hash.update(str(self.config_list).encode("latin"))
        if force or self._model_fitted_hash != parameter_hash:
            self.surrogate_model.fit(self.X, self.y_list)

        self._model_fitted_hash = parameter_hash

    def sample(self, n_points: int = 1, *, show_progress=False):
        super(BayesianOptimizationSampler, self).sample(n_points=n_points, show_progress=show_progress)
        # Make sure at end of sampling surrogate is fitted
        self.fit_surrogate()

    def _sample(self, n_points: int = 1, pbar: Union[ProgressDummy, tqdm] = ProgressDummy()):
        # Sample initial random points if not already done or given
        self.logger.info(f"Sample {n_points} new points")
        already_sampled = 0
        current_points = len(self)
        if current_points < self.initial_points:
            self._sample_initial_points(n_points)
            already_sampled = len(self) - current_points
            pbar.update(already_sampled)
            pbar.refresh()

        # Update surrogate model
        self.fit_surrogate()

        for i in range(n_points - already_sampled):
            # select next point
            self.acq_func.update(self.incumbent[1])
            new_best_candidate = self.acq_func.get_optimum()
            new_y = self.obj_func(**new_best_candidate)

            # add new point
            self.config_list.append(new_best_candidate)
            self.y_list.append(new_y)

            # Update surrogate model
            self.fit_surrogate()
            pbar.update(1)
            pbar.refresh()

    def surrogate_score(self, configs: Union[np.ndarray, List[CS.Configuration]]) -> Tuple[np.ndarray, np.ndarray]:
        if len(configs) == 0:
            warnings.warn("No configs provided. Returning empty surrogate scores")
            return np.asarray([]), np.asarray([])

        if isinstance(configs, list) and isinstance(configs[0], CS.Configuration):
            configs = [config.get_array() for config in configs]

        x = np.asarray(configs)
        return self.surrogate_model.predict(x, return_std=True)
