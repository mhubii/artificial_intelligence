import collections
import numpy as np
from helpers import SquareGrid


class PriorityQueue:
    ###
    # Exercise: implement PriorityQueue
    ###
    def __init__(self):
        # set up a dict
        self.queue = dict()

    def empty(self):
        """
        :return: True if the queue is empty, False otherwise.
        """
        # bool(dict) returns true  if not empty
        if bool(self.queue):
            return False
        else:
            return True

    def add(self, item, priority):
        """
        Add item to the queue
        :param item: any object
        :param priority: int
        """
        # add item to queue
        self.queue[item] = priority

    def pop(self):
        """
        Get the item with the minimal priority and remove it from the queue.
        :return: item with the minimal priority
        """
        # find key of minimum priority
        key = min(self.queue, key=self.queue.get)
        del self.queue[key]
        return key


def heuristic(node_a, node_b):
    """
    Heuristic
    :param node_a: pair, (x_a, y_a)
    :param node_b: pair, (x_b, y_b)
    :return: estimated distance between node_a and node_b
    """
    ###
    # Exercise: implement a heuristic for A* search
    ###
    distance = np.linalg.norm(np.subtract(node_a, node_b))
    return distance


def a_star_search(graph, start, goal):
    """

    :param graph: SquareGrid, defines the graph where we build a route.
    :param start: pair, start node coordinates (x, y)
    :param goal: pair, goal node coordinates(x, y)
    :return:
        came_from: dict, with keys - coordinates of the nodes. came_from[X] is coordinates of
            the node from which the node X was reached.
            This dict will be used to restore final path.
        cost_so_far: dict,
    """
    came_from = dict()
    cost_so_far = dict()
    ###
    # Exercise: implement A* search
    ###
    # initialize came_from with start and cost with 0
    came_from[start] = start
    cost_so_far[start] = 0

    # create an instance of PriorityQueue and add first two known nodes
    q = PriorityQueue()
    q.add(start, heuristic(start, goal))

    # expanded nodes
    expanded = []

    searching = True

    # search till goal reached
    while searching:
        # take cheapest possible step
        current_node = q.pop()

        expanded.append(current_node)

        # update cost_so_far
        cost_so_far[current_node] = cost_so_far[came_from[current_node]] + \
                                    heuristic(came_from[current_node], current_node)

        if current_node == goal:
            searching = False

        # frontier holds feasible neighbors
        frontier = graph.neighbors(current_node)

        # add frontier to PriorityQueue
        for node in frontier:
            cost = cost_so_far[current_node] + heuristic(current_node, node) + heuristic(node, goal)

            if node in expanded:
                continue
            elif node in q.queue and q.queue[node] > cost:
                came_from[node] = current_node
                q.add(node, cost)
            elif node in q.queue and q.queue[node] < cost:
                continue
            else:
                came_from[node] = current_node
                q.add(node, cost)

        if q.empty():
            searching = False
            cost_so_far[current_node] = -1

    return came_from, cost_so_far


def reconstruct_path(came_from, start, goal):
    """
    Reconstruct path using came_from dict
    :param came_from: ict, with keys - coordinates of the nodes. came_from[X] is coordinates of
            the node from which the node X was reached.
    :param start: pair, start node coordinates (x, y)
    :param goal: pair, goal node coordinates(x, y)
    :return: path: list, contains coordinates of nodes in the path

    """
    current = goal
    path = []
    if goal not in came_from:
        return path
    while current != start:
        current = came_from[current]
        path.append(current)
    path.append(start)
    path.reverse()
    return path
