from .graph_problem_interface import *
from .astar import AStar
from typing import Optional, Callable
import numpy as np
import math


class AStarEpsilon(AStar):
    """
    This class implements the (weighted) A*Epsilon search algorithm.
    A*Epsilon algorithm basically works like the A* algorithm, but with
    another way to choose the next node to expand from the open queue.
    """

    solver_name = 'A*eps'

    def __init__(self,
                 heuristic_function_type: HeuristicFunctionType,
                 within_focal_priority_function: Callable[[SearchNode, GraphProblem, 'AStarEpsilon'], float],
                 heuristic_weight: float = 0.5,
                 max_nr_states_to_expand: Optional[int] = None,
                 focal_epsilon: float = 0.1,
                 max_focal_size: Optional[int] = None):
        # A* is a graph search algorithm. Hence, we use close set.
        super(AStarEpsilon, self).__init__(heuristic_function_type, heuristic_weight,
                                           max_nr_states_to_expand=max_nr_states_to_expand)
        self.focal_epsilon = focal_epsilon
        if focal_epsilon < 0:
            raise ValueError(f'The argument `focal_epsilon` for A*eps should be >= 0; '
                             f'given focal_epsilon={focal_epsilon}.')
        self.within_focal_priority_function = within_focal_priority_function
        self.max_focal_size = max_focal_size

    def _init_solver(self, problem):
        super(AStarEpsilon, self)._init_solver(problem)

    def _extract_next_search_node_to_expand(self, problem: GraphProblem) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         by focusing on the current FOCAL and choosing the node
         with the best within_focal_priority from it.
        TODO [Ex.38]: Implement this method!
        Find the minimum expanding-priority value in the `open` queue.
        Calculate the maximum expanding-priority of the FOCAL, which is
         the min expanding-priority in open multiplied by (1 + eps) where
         eps is stored under `self.focal_epsilon`.
        Create the FOCAL by popping items from the `open` queue and inserting
         them into a focal list. Don't forget to satisfy the constraint of
         `self.max_focal_size` if it is set (not None).
        Notice: You might want to pop items from the `open` priority queue,
         and then choose an item out of these popped items. Don't forget:
         the other items have to be pushed back into open.
        Inspect the base class `BestFirstSearch` to retrieve the type of
         the field `open`. Then find the definition of this type and find
         the right methods to use (you might want to peek the head node, to
         pop/push nodes and to query whether the queue is empty).
        For each node (candidate) in the created focal, calculate its priority
         by callingthe function `self.within_focal_priority_function` on it.
         This function expects to get 3 values: the node, the problem and the
         solver (self). You can create an array of these priority value. Then,
         use `np.argmin()` to find the index of the item (within this array)
         with the minimal value. After having this index you could pop this
         item from the focal (at this index). This is the node that have to
         be eventually returned.
        Don't forget to handle correctly corner-case like when the open queue
         is empty. In this case a value of `None` has to be returned.
        Note: All the nodes that were in the open queue at the beginning of this
         method should be kept in the open queue at the end of this method, except
         for the extracted (and returned) node.
        """
        # TODO: open order
        if self.open.is_empty():
            return None

        # calculate the min f_score of nodes in open
        priority_list = []
        for i in range(len(self.open)):
            node = self.open.pop_next_node()
            priority_list.append(self._calc_node_expanding_priority(node))
            self.open.push_node(node)
        max_expanding_priority = min(priority_list) * (1 + self.focal_epsilon)

        # creates the focal_list
        focal_list = []
        focal_len = min(len(self.open), self.max_focal_size)
        if self.max_focal_size is None:
            focal_len = len(self.open)

        for i in range(focal_len):
            node = self.open.pop_next_node()
            if node.expanding_priority <= max_expanding_priority:
                focal_list.append(node)
            else:
                self.open.push_node(node)

        # calculate the best_nodes list, and choose one of its node to be the node to expand
        h_focal_priority = np.array([self.within_focal_priority_function(n, problem, self) for n in focal_list])
        node_to_expand_index = h_focal_priority.argmin()
        node_to_expand = focal_list[node_to_expand_index]

        # push back the unchoosen nodes to open
        for elem in focal_list:
            if node_to_expand != elem:
                self.open.push_node(elem)

        # push the node_to_expand to close
        if self.use_close:
            self.close.add_node(node_to_expand)
        return node_to_expand




