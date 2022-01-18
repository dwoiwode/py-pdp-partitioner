import abc
from typing import List, Tuple, Optional

import ConfigSpace as CS
import numpy as np

Sample = Tuple[np.ndarray, np.ndarray]  # configurations, variances


class AbstractPartitioner(abc.ABC):
    def __init__(self):
        pass



class DTNode:
    def __init__(self, parent: Optional["DTNode"], index_arr: np.ndarray, depth: int, max_depth: int):
        self.parent = parent
        self.index_arr = index_arr  # matrix determining samples included in split
        self.depth = depth
        self.max_depth = max_depth

        self.left_child: Optional[DTNode] = None  # <= split_value
        self.right_child: Optional[DTNode] = None  # > split_value
        self.split_value: Optional[float] = None  # float
        self.split_indices: Optional[Tuple[int, int]] = None  # t, j
        self.loss_val: float = Optional[None]  # l2 loss regarding variance impurity

    def __contains__(self, item: CS.Configuration) -> bool:
        if self.is_root():
            return True
        config_array = item.get_array()

        parent_split_value = self.parent.split_value
        we_are_left_child = self.parent.left_child == self
        if we_are_left_child:
            return config_array[self.parent.split_indices[1]] <= parent_split_value and (item in self.parent)
        else:  # we are right_child
            return config_array[self.parent.split_indices[1]] > parent_split_value and (item in self.parent)

    def is_terminal(self) -> bool:
        # either max depth or single instance
        return self.depth >= self.max_depth or np.sum(self.index_arr) == 1

    def is_root(self) -> bool:
        return self.parent is None

    def filter_pdp(self, x_ice, y_ice, variances):
        return (
            np.mean(x_ice[self.index_arr], axis=0),
            y_ice[self.index_arr].T,
            variances[self.index_arr],
        )


class DecisionTreePartitioner(AbstractPartitioner):
    def __init__(self, idx: int, x_ice: np.ndarray, variances: np.ndarray):
        super().__init__()
        self.idx = idx
        self.x = x_ice
        self.variances = variances

        assert len(x_ice.shape) == 3, 'x needs to be 3-dimensional'  # 1 feat. selected
        assert x_ice.shape[2] > 1, 'x needs at least one feature to split on'
        self.num_instances = x_ice.shape[0]
        self.num_grid_points = x_ice.shape[1]
        self.num_features = x_ice.shape[2]
        self.possible_split_params = list(set(range(self.num_features)) - {idx})
        self.root:Optional[DTNode] = None
        self.leaves:List[DTNode] = []

    def partition(self, max_depth: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        assert max_depth > 0, f'Cannot split partition for depth < 1, but got {max_depth}'

        # create root node
        index_arr = np.ones((self.num_instances,), dtype=bool)
        self.root = DTNode(None, index_arr, depth=0, max_depth=max_depth)
        self.leaves = []

        queue = [self.root]
        while len(queue) > 0:
            node = queue.pop()

            # calculate children
            left_child, right_child = self.calc_best_split(node)
            if not left_child.is_terminal():
                queue.append(left_child)
            else:
                self.leaves.append(left_child)
            if not right_child.is_terminal():
                queue.append(right_child)
            else:
                self.leaves.append(right_child)

        # calculate mean variance per partition
        partition_means = np.zeros((len(self.leaves),), dtype=float)
        partition_indices = np.zeros((len(self.leaves), self.num_instances), dtype=bool)
        for i, node in enumerate(self.leaves):
            partition_means[i] = self.calc_partition_mean(node)
            partition_indices[i] = node.index_arr

        # sort according to mean variance
        order = np.argsort(partition_means)
        partition_means = partition_means[order]
        partition_indices = partition_indices[order]

        return partition_indices, partition_means

    def calc_best_split(self, node: DTNode) -> Tuple[DTNode, DTNode]:
        best_j = -1
        best_t = -1
        best_loss = np.inf
        for j in self.possible_split_params:
            for t in range(self.num_instances):
                # get children after split
                left_indices, right_indices = self.calc_children_indices(node, j, t)

                # calculate loss
                left_loss = self.calc_loss(left_indices)
                right_loss = self.calc_loss(right_indices)
                loss = left_loss + right_loss

                # update if necessary
                if loss < best_loss:
                    best_j = j
                    best_t = t
                    best_loss = loss

        # split according to best values
        left_indices, right_indices = self.calc_children_indices(node, best_j, best_t)
        left_child = DTNode(node, left_indices, node.depth + 1, node.max_depth)
        right_child = DTNode(node, right_indices, node.depth + 1, node.max_depth)
        left_child.loss_val = self.calc_loss(left_indices)
        right_child.loss_val = self.calc_loss(right_indices)

        # update attributes of parent node
        node.left_child = left_child
        node.right_child = right_child
        node.split_value = self.x[best_t, 0, best_j]
        node.split_indices = (best_t, best_j)
        node.loss_val = best_loss

        return left_child, right_child

    def calc_children_indices(self, node, j, t) -> Tuple[np.ndarray, np.ndarray]:
        # get split values
        split_val = self.x[t, 0, j]  # index in second dim does not matter
        instance_vals = self.x[:, 0, j]
        split_cond = (instance_vals <= split_val)

        # indices after split
        left_indices = np.copy(node.index_arr)
        left_indices[~split_cond] = 0
        right_indices = np.copy(node.index_arr)
        right_indices[split_cond] = 0

        return left_indices, right_indices

    def calc_loss(self, indices: np.ndarray) -> float:
        # l2 loss calculation according to paper
        variance_grid_points = self.variances[indices, :]
        mean_variances = np.mean(variance_grid_points, axis=0)

        pointwise_l2_loss = (variance_grid_points - mean_variances) ** 2
        # pointwise_l2_loss = (self.variances - mean_variances) ** 2
        loss_sum = np.sum(pointwise_l2_loss, axis=None)

        return loss_sum.item()

    def calc_partition_mean(self, node) -> float:
        variances_in_partition = self.variances[node.index_arr]
        return np.mean(variances_in_partition, axis=None).item()


class RandomForestPartitioner(AbstractPartitioner):
    pass
