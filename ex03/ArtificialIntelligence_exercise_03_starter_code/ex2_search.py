import collections
import math


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
        # not bool(dict) returns true  if empty
        return not bool(self.queue)

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
    distance = math.sqrt(abs(node_a[0] - node_b[0]) + abs(node_a[1] - node_b[1]))
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
    # create an instance of PriorityQueue and add first two known nodes
    q = PriorityQueue()
    q.add(start, heuristic(start, goal))

    came_from[start] = None
    cost_so_far[start] = 0

    # search till goal reached
    while not q.empty():
        # take cheapest possible step
        current = q.pop()

        if current == goal:
            break

        # update frontier
        for next in graph.neighbors(current):
            h = heuristic(next, goal)
            g = cost_so_far[current] + heuristic(current, next)
            f = h + g

            if next not in cost_so_far or f < g:
                cost_so_far[next] = g
                came_from[next] = current
                q.add(next, g)

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
