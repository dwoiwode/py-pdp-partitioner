import itertools
from typing import Optional, Iterable, List

import numpy as np
from matplotlib import pyplot as plt

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from pyPDP.algorithms.ice import ICE
from pyPDP.algorithms.partitioner import Partitioner, Region
from pyPDP.algorithms.partitioner.decision_tree_partitioner import DecisionTreePartitioner
from pyPDP.surrogate_models import SurrogateModel
from pyPDP.utils.plotting import get_random_color, get_ax
from pyPDP.utils.typing import SelectedHyperparameterType, ColorType


class RandomForestPartitioner(Partitioner):

    def __init__(self,
                 surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 samples: np.ndarray,
                 num_grid_points_per_axis: int = 20,
                 num_splits_per_axis: int = 100,  # number of points to test for possible split
                 min_points_per_node: int = 10,  # minimum number of ice curves in a single node
                 seed=None
                 ):
        super().__init__(
            surrogate_model=surrogate_model,
            selected_hyperparameter=selected_hyperparameter,
            num_grid_points_per_axis=num_grid_points_per_axis,
            samples=samples,
            seed=seed
        )

        self.num_splits_per_axis = num_splits_per_axis
        self.min_points_per_node = min_points_per_node
        self.seed = seed

        self.rng = np.random.default_rng(seed=seed)
        self.trees: Optional[List[DecisionTreePartitioner]] = None

    @classmethod
    def from_ICE(cls, ice: ICE, seed=None) -> "RandomForestPartitioner":
        partitioner = RandomForestPartitioner(
            surrogate_model=ice.surrogate_model,
            selected_hyperparameter=ice.selected_hyperparameter,
            samples=ice.samples,
            num_grid_points_per_axis=ice.num_grid_points_per_axis,
            seed=seed
        )
        partitioner._ice = ice
        return partitioner

    def partition(self,
                  num_trees: int = 10,
                  max_depth: int = 1,
                  sample_size: Optional[int] = None):
        assert num_trees > 0, 'Need at least one tree for splitting'
        assert sample_size > 0, 'Need at least one sample for splitting'
        assert max_depth > 0, 'Need to perform at least one split'

        if sample_size is None:
            sample_size = len(self.samples)

        # reset trees from possible last split
        self.trees = []

        for tree_index in range(num_trees):
            # calculate subset of ice curves for tree
            subset_indices = self.rng.integers(0, len(self.samples), size=(sample_size,))
            subset_samples = self.samples[subset_indices]
            subset_ice = ICE(
                self.ice.surrogate_model,
                self.ice.selected_hyperparameter,
                subset_samples,
                self.ice.num_grid_points_per_axis,
                self.seed
            )

            # calculate features subset
            n_params = len(self.possible_split_parameters)
            subset_mask = self.rng.integers(0, 2, size=(n_params,))
            splittable_hp = list(itertools.compress(self.possible_split_parameters, subset_mask))
            if len(splittable_hp) == 0:  # we need at least a single split param
                splittable_hp = [self.rng.choice(self.possible_split_parameters, size=1).item()]
            not_splittable_hp = list(set(self.possible_split_parameters) - set(splittable_hp))

            # create dt
            dt = DecisionTreePartitioner.from_ICE(subset_ice, min_points_per_node=1, not_splittable_hp=not_splittable_hp)
            dt.partition(max_depth=max_depth)
            self.trees.append(dt)

    def get_incumbent_region(self, incumbent: CS.Configuration, min_incumbent_overlap: Optional[int] = None) -> Region:
        assert self.trees is not None and len(self.trees) > 0, 'Need to partition before computing incumbent region'
        if min_incumbent_overlap is None:
            min_incumbent_overlap = len(self.trees)

        # get incumbent regions
        incumbent_regions = [dt.get_incumbent_region(incumbent) for dt in self.trees]

        # all points in at least min_incumbent_overlap are in resulting region
        result_indices = []
        for idx, sample in enumerate(self.samples):
            sample_config = CS.Configuration(self.config_space, vector=sample)
            contained_regions = [1 for region in incumbent_regions if sample_config in region]
            if sum(contained_regions) >= min_incumbent_overlap:
                result_indices.append(idx)

        result_x_ice = self.ice.x_ice[result_indices]
        result_y_ice = self.ice.y_ice[result_indices]
        result_y_variances = self.ice.y_variances[result_indices]

        result_region = Region(
            result_x_ice,
            result_y_ice,
            result_y_variances,
            self.config_space,
            self.selected_hyperparameter
        )

        return result_region

    def plot(self, *args, x_hyperparameters: Optional[Iterable[CSH.Hyperparameter]] = None,
             ax: Optional[plt.Axes] = None):
        pass

    def plot_incumbent_regions(
            self,
            incumbent: CS.Configuration,
            color_list: Optional[List[ColorType]] = None,
            alpha: float = 0.1,
            ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)
        if color_list is None:
            color_list = [get_random_color() for _ in self.trees]

        for tree, color in zip(self.trees, color_list):
            tree.plot_incumbent_cs(incumbent, color=color, ax=ax, alpha=alpha)

