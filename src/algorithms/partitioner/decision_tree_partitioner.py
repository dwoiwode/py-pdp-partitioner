from typing import List, Tuple, Optional, Any

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt

from src.algorithms.ice import ICE
from src.algorithms.partitioner import Region, Partitioner
from src.surrogate_models import SurrogateModel
from src.utils.plotting import Plottable, get_ax, check_and_set_axis, get_random_color
from src.utils.typing import SelectedHyperparameterType, ColorType
from src.utils.utils import scale_float, unscale_float, unscale


class SplitCondition:
    def __init__(self,
                 config_space: CS.ConfigurationSpace,
                 hp: CSH.Hyperparameter,
                 value: Optional[Any] = None,
                 normalized_value: Optional[Any] = None,
                 less_equal: Optional[bool] = None):

        assert value is not None or normalized_value is not None, 'Either value or normalized value has to be specified'
        assert value is None or normalized_value is None, 'Only one of value or normalized value should be given'
        if normalized_value is None:
            normalized_value = scale_float(value, config_space, hp)

        assert not isinstance(normalized_value, float) or less_equal is not None, 'floating values need less_equal'

        self.config_space = config_space
        self.hyperparameter = hp
        self._normalized_value = normalized_value
        self.less_equal = less_equal

    @property
    def normalized_value(self):
        return self._normalized_value

    @property
    def value(self):
        if isinstance(self._normalized_value, float):
            return unscale_float(self._normalized_value, self.config_space, self.hyperparameter)
        else:
            return self._normalized_value

    def is_satisfied(self, configuration: CS.Configuration) -> bool:
        config_value = configuration.get(self.hyperparameter.name)
        if isinstance(self.value, float):
            if self.less_equal:
                return config_value <= self.value
            else:
                return config_value > self.value
        else:
            return self.value == config_value

    def __str__(self):
        if self.less_equal is not None:
            if self.less_equal:
                op_str = '>'
            else:
                op_str = '<='
        else:
            op_str = 'in'
        return f'SplitCondition({self.value} {op_str} {self.hyperparameter.name})'


class DTRegion(Region, Plottable):
    def __init__(self,
                 x_points: np.ndarray,
                 y_points: np.ndarray,
                 y_variances: np.ndarray,
                 split_conditions: List[SplitCondition],
                 full_config_space: CS.ConfigurationSpace,
                 selected_hyperparameter: SelectedHyperparameterType):
        Region.__init__(self, x_points, y_points, y_variances, full_config_space, selected_hyperparameter)
        Plottable.__init__(self)

        self.split_conditions = split_conditions
        self.full_config_space = full_config_space
        self.selected_hyperparameter = list(selected_hyperparameter)

    def __contains__(self, item: CS.Configuration) -> bool:
        for condition in self.split_conditions:
            if not condition.is_satisfied(item):
                return False
        return True

    def implied_config_space(self, seed: int) -> CS.ConfigurationSpace:
        # copy cs
        hp_dic = {}
        for hp in self.full_config_space.get_hyperparameters():
            if isinstance(hp, CSH.NumericalHyperparameter):
                new_hp = CSH.UniformFloatHyperparameter(hp.name, lower=hp.lower, upper=hp.upper, log=hp.log)
                hp_dic[hp.name] = new_hp
            else:
                raise NotImplementedError()

        # adjust upper and lower of new cs
        for cond in self.split_conditions:
            hp = hp_dic[cond.hyperparameter.name]
            if isinstance(cond.value, float):
                if cond.less_equal and cond.value < hp.upper:
                    hp.upper = cond.value
                    hp.default_value = hp.lower
                elif not cond.less_equal and cond.value > hp.lower:
                    hp.lower = cond.value
                    hp.default_value = hp.lower
            else:
                raise NotImplementedError

        # add new hp to new cs
        cs = CS.ConfigurationSpace(seed=seed)
        for hp in hp_dic.values():
            cs.add_hyperparameter(hp)

        return cs

    def filter_by_condition(self, condition: SplitCondition) -> "DTRegion":
        hyperparameter_idx = self.full_config_space.get_idx_by_hyperparameter_name(condition.hyperparameter.name)
        instance_vals_at_idx = self.x_points[:, 0, hyperparameter_idx]  # second dimension does not matter
        if condition.less_equal:
            func_split_cond = (instance_vals_at_idx <= condition.normalized_value)
        else:
            func_split_cond = (instance_vals_at_idx > condition.normalized_value)

        new_x_points = np.copy(self.x_points[func_split_cond])
        new_y_points = np.copy(self.y_points[func_split_cond])
        new_y_variances = np.copy(self.y_variances[func_split_cond])
        new_conditions = self.split_conditions + [condition]
        new_region = DTRegion(new_x_points, new_y_points, new_y_variances, new_conditions, self.full_config_space,
                              self.selected_hyperparameter)

        return new_region

    def plot(self,
             color: ColorType = 'red',
             alpha: float = 0.1,
             ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)
        check_and_set_axis(ax, self.selected_hyperparameter)
        n_selected_hyperparameter = len(self.selected_hyperparameter)

        # Plot
        if n_selected_hyperparameter == 1:  # 1D
            x_unscaled = unscale(self.x_points, self.full_config_space)
            hp = self.selected_hyperparameter[0]
            hp_idx = self.full_config_space.get_idx_by_hyperparameter_name(hp.name)

            ax.plot(x_unscaled[:, :, hp_idx].T, self.y_points.T, alpha=alpha, color=color)
        elif n_selected_hyperparameter == 2:  # 2D
            raise NotImplementedError("2D currently not implemented (#TODO)")
        else:
            raise NotImplementedError(f"Plotting for {n_selected_hyperparameter} dimensions not implemented. "
                                      "Please select a specific hp by setting `selected_hyperparameters`")


class DTNode:
    def __init__(self,
                 parent: Optional["DTNode"],
                 region: DTRegion,
                 depth: int,
                 config_space: CS.ConfigurationSpace,
                 selected_hyperparameter: SelectedHyperparameterType):
        self.parent = parent
        self.region = region
        self.depth = depth
        self.config_space = config_space
        self.selected_hyperparameter = selected_hyperparameter

        self.left_child: Optional[DTNode] = None  # <= split_value
        self.right_child: Optional[DTNode] = None  # > split_value

    def __contains__(self, item: CS.Configuration) -> bool:
        return item in self.region

    def __len__(self):
        return len(self.region)

    def is_splittable(self) -> bool:
        return len(self.region) > 1

    def is_root(self) -> bool:
        return self.parent is None

    def split_at_idx(self, hyperparameter: CSH.Hyperparameter, instance_idx: int) -> Tuple["DTNode", "DTNode"]:
        hyperparameter_idx = self.config_space.get_idx_by_hyperparameter_name(hyperparameter.name)
        split_val = self.region.x_points[instance_idx, 0, hyperparameter_idx]  # index in second dim does not matter

        left_split_condition = SplitCondition(self.config_space, hyperparameter, normalized_value=split_val,
                                              less_equal=True)
        left_region = self.region.filter_by_condition(left_split_condition)
        left_node = DTNode(self, left_region, self.depth + 1, self.config_space,
                           self.selected_hyperparameter)

        right_split_condition = SplitCondition(self.config_space, hyperparameter, normalized_value=split_val,
                                               less_equal=False)
        right_region = self.region.filter_by_condition(right_split_condition)
        right_node = DTNode(self, right_region, self.depth + 1, self.config_space,
                            self.selected_hyperparameter)

        return left_node, right_node


class DTPartitioner(Partitioner):
    def __init__(self,
                 surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 num_grid_points_per_axis: int = 20,
                 num_samples: int = 1000,
                 ):
        super().__init__(surrogate_model=surrogate_model,
                         selected_hyperparameter=selected_hyperparameter,
                         num_grid_points=num_grid_points_per_axis,
                         num_samples=num_samples)

        self.root: Optional[DTNode] = None
        self.leaves: List[DTNode] = []

    @classmethod
    def from_ICE(cls, ice: ICE) -> "DTPartitioner":
        partitioner = DTPartitioner(ice.surrogate_model,
                                    ice.selected_hyperparameter,
                                    ice.num_samples,
                                    ice.num_grid_points_per_axis)
        partitioner._ice = ice
        return partitioner

    def partition(self, max_depth: int = 1) -> List[DTRegion]:
        assert max_depth > 0, 'Can only split for depth > 0'

        # create root node and leaves
        dt_region = DTRegion(self.ice.x_ice, self.ice.y_ice, self.ice.y_variances, [], self.config_space,
                             self.selected_hyperparameter)
        self.root = DTNode(None, dt_region, depth=0, config_space=self.config_space,
                           selected_hyperparameter=self.selected_hyperparameter)
        self.leaves = []

        queue = [self.root]
        while len(queue) > 0:
            node = queue.pop()
            if not node.is_splittable() or node.depth >= max_depth:
                self.leaves.append(node)
                continue

            # calculate children
            queue += self.calc_best_split(node)

        leaf_regions = [leaf.region for leaf in self.leaves]
        return leaf_regions

    def get_incumbent_region(self, incumbent: CS.Configuration) -> DTRegion:
        assert self.leaves is not None and len(self.leaves) > 0, 'Cannot compute incumbent region before partitioning'
        for leaf in self.leaves:
            if incumbent in leaf:
                return leaf.region

    def calc_best_split(self, node: DTNode) -> Tuple[DTNode, DTNode]:
        assert node.is_splittable(), 'Cannot split a terminal node'

        best_loss = np.inf
        best_left_child = None
        best_right_child = None
        for hyperparameter in self.possible_split_parameters:
            for instance_idx in range(len(node)):
                # get children after split
                left_child, right_child = node.split_at_idx(hyperparameter, instance_idx)

                # calculate loss
                left_loss = left_child.region.loss
                right_loss = right_child.region.loss
                total_loss = left_loss + right_loss

                # update best values if necessary
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_left_child = left_child
                    best_right_child = right_child

        assert best_left_child is not None and best_right_child is not None

        # update attributes of parent node
        node.left_child = best_left_child
        node.right_child = best_right_child

        return best_left_child, best_right_child

    def plot(self,
             color_list: Optional[List[ColorType]] = None,
             alpha: float = 0.1,
             ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)
        assert self.leaves is not None and len(self.leaves) != 0, 'Please call partition before plotting'

        if color_list is None:
            color_list = [get_random_color() for _ in self.leaves]
        assert len(color_list) == len(self.leaves), 'Color list needs a color for every leaf'

        for i, leaf in enumerate(self.leaves):
            leaf.region.plot(color=color_list[i], alpha=alpha, ax=ax)
