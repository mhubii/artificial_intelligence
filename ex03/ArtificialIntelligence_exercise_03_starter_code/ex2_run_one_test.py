from __future__ import print_function
import time

from helpers import *
from ex2_search import *


if __name__ == '__main__':
    grid, start, goal, cost, time_limit = load_test('grids/test_06.txt')
    start_time = time.time()
    came_from, cost_so_far = a_star_search(grid, start, goal)
    duration = time.time() - start_time
    is_ok = duration <= time_limit
    print('A* Elapsed time : {}s ({})'.format(duration, 'OK' if is_ok else 'Time limit exceeded'))
    print('Field')
    draw_grid(grid, width=3, point_to=came_from, start=start, goal=goal)
    print('Cost:')
    draw_grid(grid, width=3, number=cost_so_far, start=start, goal=goal)
    print('Path:')
    draw_grid(grid, width=3, path=reconstruct_path(came_from, start=start, goal=goal))
