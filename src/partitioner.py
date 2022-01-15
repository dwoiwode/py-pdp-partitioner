import abc
from typing import List, Tuple

import ConfigSpace as CS
import numpy as np

Sample = Tuple[np.ndarray, np.ndarray]  # configurations, variances

class AbstractPartitioner(abc.ABC):
    def __init__(self):
        pass

    # @abc.abstractmethod
    # def partition(self):
    #     pass

class DTNode:
    def __init__(self, index_arr: np.ndarray, depth: int, max_depth: int):
        self.index_arr = index_arr  # matrix determining samples included in split
        self.depth = depth
        self.max_depth = max_depth

        self.left_child = None
        self.right_child = None
        self.split_value = None  # t
        self.split_feature_index = None  # j
        self.loss_val = None  # l2 loss regarding variance impurity

    def is_terminal(self):
        # either max depth or single instance
        return self.depth >= self.max_depth or np.sum(self.index_arr) == 1


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
        self.root = None

    def partition(self, max_depth: int = 1):
        assert max_depth > 0, f'Cannot split partition for depth < 1, but got {max_depth}'

        # create root node
        index_arr = np.ones((self.num_instances,), dtype=bool)
        root = DTNode(index_arr, depth=0, max_depth=max_depth)

        queue = [root]
        while len(queue) > 0:
            node = queue.pop()

            # calculate children
            left_child, right_child = self.calc_best_split(node)
            if left_child.depth >= max_depth:
                left_child.is_terminal = True
                right_child.is_terminal = True
            else:
                queue.append(node.left_child)
                queue.append(node.right_child)

        self.root = root

    def calc_best_split(self, node: DTNode) -> Tuple[DTNode, DTNode]:
        best_j = -1
        best_t = -1
        best_loss = np.inf
        for j in self.possible_split_params:
            for t in range(self.num_instances):
                # get children after split
                left_indices, right_indices = self.calc_children(node, j, t)

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
        left_indices, right_indices = self.calc_children(node, best_j, best_t)
        left_child = DTNode(left_indices, node.depth + 1, node.max_depth)
        right_child = DTNode(right_indices, node.depth + 1, node.max_depth)
        left_child.loss_val = self.calc_loss(left_indices)
        right_child.loss_val = self.calc_loss(right_indices)

        # update attributes of parent node
        node.left_child = left_child
        node.right_child = right_child
        node.split_value = best_t
        node.split_feature_index = best_j
        node.loss_val = best_loss

        return left_child, right_child

    def calc_children(self, node, j, t):
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
        loss_sum = np.sum(pointwise_l2_loss, axis=None)

        return loss_sum.item()





class RandomForestPartitioner(AbstractPartitioner):
    pass
