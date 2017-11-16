import numpy as np
import random
import matplotlib.pyplot as plt


def f(x):
    """
    Energy function.
    """
    return 0.2 * np.sin(12.5 * x) + (x - 1)**2 - 5


def p(y_current, y_next, T):
    """
    Probability.
    """
    return np.exp(-(y_next-y_current)/T)


def minimize(x, y, start_position):
    """
    Minimize the energy function.
    :param x: array, x coordinates
    :param y: array, values of the energy function
    :param start_position: int, initial position of the agent
    :return: position with the minimal found value of energy function
    """
    best_pos = start_position
    T = 100
    alpha = 10

    # update the best position
    for iter_num in xrange(400):
        ###
        # Exercise: implement Simulated annealing
        # update best_pos if found the better one
        ###
        # by default the algorithm walks to the right
        y_current = y[best_pos]
        y_next = y[best_pos + 1]

        if y_current < y_next:
            prob = p(y_current, y_next, alpha * T)
            choice = [best_pos, best_pos + 1]
            best_pos = np.random.choice(a=choice, size=1, p=[1-prob, prob])[0]
        else:
            best_pos += 1

        alpha /= np.log(iter_num+2)

    assert 0 <= best_pos < len(y), 'incorrect index'
    return best_pos


def main():
    random.seed(2017)
    np.random.seed(2017)
    x = np.linspace(-0.5, 2, num=31, endpoint=True)
    y = f(x)
    print 'x = %s' % x
    print 'y = %s' % y
    start_position = 0

    best_pos = minimize(x, y, start_position)
    print "Best value %s at pos %d" % (y[best_pos], best_pos)

    if best_pos != np.argmin(y):
        print 'You haven\'t found the global minimum. Try harder!'
    else:
        print 'Success!'

    plt.plot(x, y)
    plt.plot(x, y, '--')
    plt.plot(x[start_position], y[start_position], '-bo', label='start pos', markersize=13)
    plt.plot(x[best_pos], y[best_pos], '-go', label='best found pos', markersize=13)
    plt.title('f(x)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
