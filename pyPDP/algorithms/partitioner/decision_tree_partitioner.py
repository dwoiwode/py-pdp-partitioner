from typing import List, Tuple, Optional, Any

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from matplotlib import pyplot as plt

from pyPDP.algorithms.ice import ICE
from pyPDP.algorithms.partitioner import Region, Partitioner
from pyPDP.surrogate_models import SurrogateModel
from pyPDP.utils.plotting import get_ax, check_and_set_axis, get_random_color, plot_config_space
from pyPDP.utils.typing import SelectedHyperparameterType, ColorType
from pyPDP.utils.utils import scale_float, unscale_float, unscale, ConfigSpaceHolder, get_hyperparameters


class SplitCondition(ConfigSpaceHolder):
    def __init__(self,
                 config_space: CS.ConfigurationSpace,
                 hp: CSH.Hyperparameter,
                 value: Optional[Any] = None,
                 normalized_value: Optional[Any] = None,
                 less_equal: Optional[bool] = None):
        super().__init__(config_space)
        assert value is not None or normalized_value is not None, 'Either value or normalized value has to be specified'
        assert value is None or normalized_value is None, 'Only one of value or normalized value should be given'
        if normalized_value is None:
            normalized_value = scale_float(value, self.config_space, hp)

        assert not isinstance(normalized_value, float) or less_equal is not None, 'floating values need less_equal'

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
        """
        E.g. SplitCondition(x1 > 3.0)
        """
        if self.less_equal is not None:
            if self.less_equal:
                op_str = '<='
            else:
                op_str = '>'
            return f'SplitCondition({self.hyperparameter.name} {op_str} {self.value})'
        else:
            return f'SplitCondition({self.value} in {self.hyperparameter.name})'


class DTRegion(Region):
    def __init__(self,
                 parent: Optional['DTRegion'],
                 depth: int,
                 x_points: np.ndarray,
                 y_points: np.ndarray,
                 y_variances: np.ndarray,
                 split_conditions: List[SplitCondition],
                 full_config_space: CS.ConfigurationSpace,
                 selected_hyperparameter: SelectedHyperparameterType):
        Region.__init__(self, x_points, y_points, y_variances, full_config_space, selected_hyperparameter)

        self.parent = parent
        self.depth = depth
        self.selected_hyperparameter = selected_hyperparameter

        self.left_child: Optional[DTRegion] = None  # <= split_value
        self.right_child: Optional[DTRegion] = None  # > split_value

        self.split_conditions = split_conditions

    def __contains__(self, item: CS.Configuration) -> bool:
        for condition in self.split_conditions:
            if not condition.is_satisfied(item):
                return False
        return True

    def implied_config_space(self, seed: Optional[int] = None) -> CS.ConfigurationSpace:
        # copy cs
        hp_dic = {}
        for hp in self.config_space.get_hyperparameters():
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
        hyperparameter_idx = self.config_space.get_idx_by_hyperparameter_name(condition.hyperparameter.name)
        instance_vals_at_idx = self.x_points[:, 0, hyperparameter_idx]  # second dimension does not matter
        if condition.less_equal:
            func_split_cond = (instance_vals_at_idx <= condition.normalized_value)
        else:
            func_split_cond = (instance_vals_at_idx > condition.normalized_value)

        new_x_points = np.copy(self.x_points[func_split_cond])
        new_y_points = np.copy(self.y_points[func_split_cond])
        new_y_variances = np.copy(self.y_variances[func_split_cond])
        new_conditions = self.split_conditions + [condition]
        new_region = DTRegion(
            parent=self,
            depth=self.depth + 1,
            x_points=new_x_points,
            y_points=new_y_points,
            y_variances=new_y_variances,
            split_conditions=new_conditions,
            full_config_space=self.config_space,
            selected_hyperparameter=self.selected_hyperparameter
        )

        return new_region

    def is_splittable(self) -> bool:
        return len(self) > 1

    def is_root(self) -> bool:
        return self.parent is None

    def split_at_value(self, hyperparameter: CSH.Hyperparameter, split_val: int) -> Tuple["DTRegion", "DTRegion"]:
        # Left side
        left_split_condition = SplitCondition(self.config_space, hyperparameter, normalized_value=split_val,
                                              less_equal=True)
        left_region = self.filter_by_condition(left_split_condition)

        # Right side
        right_split_condition = SplitCondition(self.config_space, hyperparameter, normalized_value=split_val,
                                               less_equal=False)
        right_region = self.filter_by_condition(right_split_condition)

        return left_region, right_region

    def plot(self,
             color: ColorType = 'red',
             alpha: float = 0.1,
             ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)
        check_and_set_axis(ax, self.selected_hyperparameter)
        n_selected_hyperparameter = len(self.selected_hyperparameter)

        # Plot
        if n_selected_hyperparameter == 1:  # 1D
            x_unscaled = unscale(self.x_points, self.config_space)
            hp = self.selected_hyperparameter[0]
            hp_idx = self.config_space.get_idx_by_hyperparameter_name(hp.name)

            ax.plot(x_unscaled[:, :, hp_idx].T, self.y_points.T, alpha=alpha, color=color)
        elif n_selected_hyperparameter == 2:  # 2D
            raise NotImplementedError("2D currently not implemented (#TODO)")
        else:
            raise NotImplementedError(f"Plotting for {n_selected_hyperparameter} dimensions not implemented. "
                                      "Please select a specific hp by setting `selected_hyperparameters`")


class DecisionTreePartitioner(Partitioner):
    def __init__(self,
                 surrogate_model: SurrogateModel,
                 selected_hyperparameter: SelectedHyperparameterType,
                 samples: np.ndarray,
                 num_grid_points_per_axis: int = 20,
                 num_splits_per_axis: int = 100,  # number of possible split points per axis
                 min_points_per_node: int = 10,  # minimum number of ice curves in a single node
                 not_splittable_hp: Optional[SelectedHyperparameterType] = None,  # more hp to ignore for splitting
                 seed=None
                 ):
        super().__init__(
            surrogate_model=surrogate_model,
            selected_hyperparameter=selected_hyperparameter,
            num_grid_points_per_axis=num_grid_points_per_axis,
            samples=samples,
            not_splittable_hp=not_splittable_hp,
            seed=seed
        )
        self.num_splits_per_axis = num_splits_per_axis
        self.min_points_per_node = min_points_per_node

        self.root: DTRegion = DTRegion(
            parent=None,
            depth=0,
            full_config_space=self.config_space,
            x_points=self.ice.x_ice,
            y_points=self.ice.y_ice,
            y_variances=self.ice.y_variances,
            split_conditions=[],
            selected_hyperparameter=self.selected_hyperparameter
        )
        self.leaves: List[DTRegion] = []

    @classmethod
    def from_ICE(cls, ice: ICE,
                 num_splits_per_axis: int = 100,
                 min_points_per_node: int = 10,
                 not_splittable_hp: Optional[SelectedHyperparameterType] = None,  # more hp to ignore for splitting
                 ) -> "DecisionTreePartitioner":
        partitioner = DecisionTreePartitioner(
            surrogate_model=ice.surrogate_model,
            selected_hyperparameter=ice.selected_hyperparameter,
            samples=ice.samples,
            num_grid_points_per_axis=ice.num_grid_points_per_axis,
            num_splits_per_axis=num_splits_per_axis,
            min_points_per_node=min_points_per_node,
            not_splittable_hp=not_splittable_hp
        )
        partitioner._ice = ice
        return partitioner

    def partition(self, max_depth: int = 1) -> List[DTRegion]:
        assert max_depth > 0, 'Can only split for depth > 0'

        # create root node and leaves
        self.leaves = []

        queue = [self.root]
        while len(queue) > 0:
            node = queue.pop()
            if not node.is_splittable() or node.depth >= max_depth:
                self.leaves.append(node)
                continue

            # calculate children
            split_result = self.calc_best_split(node)
            if split_result is None:
                self.leaves.append(node)
                continue
            queue += split_result

        return self.leaves

    def get_incumbent_region(self, incumbent: CS.Configuration) -> DTRegion:
        assert self.leaves is not None and len(self.leaves) > 0, 'Cannot compute incumbent region before partitioning'
        for leaf in self.leaves:
            if incumbent in leaf:
                return leaf

    def calc_best_split(self, node: DTRegion) -> Optional[Tuple[DTRegion, DTRegion]]:
        assert node.is_splittable(), 'Cannot split a terminal node'

        best_loss = np.inf
        best_left_child = None
        best_right_child = None
        for hyperparameter in self.possible_split_parameters:
            # split values excluding upper and lower bound
            possible_split_vals = np.linspace(0, 1, num=self.num_splits_per_axis + 2)[1:-1]
            for split_val in possible_split_vals:
                # get children after split
                left_child, right_child = node.split_at_value(hyperparameter, split_val)
                if len(left_child) < self.min_points_per_node or len(right_child) < self.min_points_per_node:
                    continue

                # calculate loss
                left_loss = left_child.loss
                right_loss = right_child.loss
                total_loss = left_loss + right_loss

                # update best values if necessary
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_left_child = left_child
                    best_right_child = right_child

        if best_left_child is None or best_right_child is None:
            return None
        # assert best_left_child is not None and best_right_child is not None

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
            leaf.plot(color=color_list[i], alpha=alpha, ax=ax)

    def plot_incumbent_cs(self,
                          incumbent: CS.Configuration,
                          color: ColorType = "orange",
                          alpha: float = 0.5,
                          ax: Optional[plt.Axes] = None):
        ax = get_ax(ax)
        region = self.get_incumbent_region(incumbent)
        new_cs = region.implied_config_space()
        all_hp = new_cs.get_hyperparameters()
        not_selected_hp = sorted(list(set(all_hp) - set(self.selected_hyperparameter)), key=lambda hp: hp.name)
        plot_config_space(new_cs, x_hyperparameters=not_selected_hp, color=color, alpha=alpha, ax=ax)
