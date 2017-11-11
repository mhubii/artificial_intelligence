from __future__ import print_function
import collections
import heapq
import numpy as np


class SquareGrid(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.walls

    def neighbors(self, id):
        (x, y) = id
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

    def cost(self, from_node, to_node):
        return 1


# utility functions for dealing with square grids
def from_id_width(id, width):
    return (id % width, id // width)


def draw_tile(graph, id, style, width):
    r = "."
    if 'number' in style and id in style['number']: r = u"%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = u"\u2192"
        if x2 == x1 - 1: r = u"\u2190"
        if y2 == y1 + 1: r = u"\u2193"
        if y2 == y1 - 1: r = u"\u2191"
    if 'start' in style and id == style['start']: r = u"A"
    if 'goal' in style and id == style['goal']: r = u"Z"
    if 'path' in style and id in style['path']: r = u"@"
    if id in graph.walls: r = u"#" * width
    return r


def draw_grid(graph, width=2, **style):
    for y in range(graph.height):
        for x in range(graph.width):
            print("%%-%ds" % width % draw_tile(graph, (x, y), style, width), end="")
        print()


def load_test(path, verbose=1):
    with open(path) as f:
        header = f.readline()
        cost = int(f.readline())
        time_limit = float(f.readline())
        start = tuple(map(int, header.split(','))[:2])
        goal = tuple(map(int, header.split(','))[2:])
        s = f.read()
        width = s.find('\n')
        assert width > 0
        cells = [x for x in s if x != '\n']
        assert len(cells) % width == 0
        height = len(cells) / width
        walls = np.nonzero(np.array(cells) == '#')[0]
        grid = SquareGrid(width, height)
        grid.walls = [from_id_width(cur_id, width=width) for cur_id in walls]
        if verbose:
            print('w x h = {}x{}'.format(width, height))
            print('start={}; goal={}'.format(start, goal))
            print('Time limit: {}s'.format(time_limit))
        return grid, start, goal, cost, time_limit
