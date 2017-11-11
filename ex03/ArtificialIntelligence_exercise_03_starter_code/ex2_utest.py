from __future__ import print_function
import time
import glob
import os
from collections import OrderedDict
import numpy as np

from helpers import load_test
from ex2_search import heuristic
from ex2_search import PriorityQueue
from ex2_search import a_star_search as a_star_search

from helpers import draw_grid


def test_heuristic():
    try:
        a = (2, 2)
        b = (10, 12)
        c = (5, 7)
        if heuristic(a, b) > 18 or (heuristic(c, b) + 8) < heuristic(a, b):
            return 1
    except:
        return 1
    return 0


def test_PriorityQueue():
    try:
        q = PriorityQueue()
        q.add('a', 10)
        q.add('b', 31)
        if q.pop() != 'a':
            return 1
        q.add('c', -2)
        q.add('d', 0)
        if q.pop() != 'c':
            return 1
        if q.empty():
            return 1
    except:
        return 1
    return 0


def print_test(name, ret_code):
    print('Test {}: [{}]'.format(name, 'OK' if ret_code == 0 else 'Failed'))


def test():
    cnt_ok = 0
    ret_code = test_heuristic()
    print_test('test_heuristic', ret_code)
    if ret_code != 0:
        return cnt_ok
    cnt_ok += 1

    ret_code = test_PriorityQueue()
    print_test('test_PriorityQueue', ret_code)
    if ret_code != 0:
        return cnt_ok
    cnt_ok += 1

    paths = sorted(glob.glob('grids/*.txt'))
    grids = OrderedDict()
    for p in paths:
        grids[os.path.basename(p).split('.')[0]] = load_test(p, verbose=0)

    for name, (grid, start, goal, cost, time_limit) in grids.iteritems():
        is_ok = True
        try:
            print('Time limit: ', time_limit, '(%s)' % name)

            start_time = time.time()
            came_from, cost_so_far = a_star_search(grid, start, goal)
            duration = time.time() - start_time
            msg = 'OK'
            if duration > time_limit:
                msg = 'Failed: Time Limit'
                is_ok = False
            elif goal in came_from:
                if cost == -1 or cost != cost_so_far[goal]:
                    msg = 'Failed: Wrong Answer'
                    is_ok = False
            elif goal not in came_from and cost != -1:
                msg = 'Failed: Wrong Answer'
                is_ok = False
            cost_got = cost_so_far[goal] if goal in cost_so_far else -1
        except:
            msg = 'Failed'
            cost_got = None
            duration = np.nan
            is_ok = False
        cnt_ok += int(is_ok)
        print('Test {}: [{}] ({:.3f}s) (cost={})'.format(name, msg, duration, cost_got))
    return cnt_ok


if __name__ == '__main__':
    cnt_ok = test()
    print('---')
    print('Tests passed: {}'.format(cnt_ok))
