import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style


def check_epsilon(epsilon, compare_func, n, dimension):
    """ Estimates the volume of a subset of the order polytope

        Creates a number of random sample points using the dirichlet
        distribution and returns the ratio of points inside the subset to the
        number of samples taken. Uses the parameter epsilon to establish
        the level of geometric balancing used to determine inclusion to the
        desired subset.

        Args:
            epsilon: Defines the interval of a hit
            compare_func: Comparison function used to determine hits
            n: Number of samples to take
            dimension: Dimension of the ambient space

        Returns:
            hit_ratio: The ratio of hits to number of samples
    """
    hits = 0
    alpha = np.ones(dimension+1)
    for i in range(n):
        x, y = np.random.dirichlet(alpha, size=2)
        z = np.array([sum([i for i in x[:k+1]]) -
                      sum([j for j in y[:k]]) for k in range(x.shape[0])])
        d = find_odd(z)
        if d is not None:
            comp_val = comp_func(z, d)
        if d is not None and 1/2-epsilon < comp_val < 1/2+epsilon:
            hits = hits + 1
    hit_ratio = hits / n
    return hit_ratio


def find_odd(x):
    """ Determines if point with any geometric balancing exists

    Checks a vector for a unique negative or positive element

    Args:
        x: A vector to be examined for a dif

    Returns
        positions: The index with a unique sign in x, or None
                   if there is no unique sign.
    """
    c = x
    d = {-1: [], 1: []}
    for j, i in enumerate(np.sign(c)):
        if i in d:
            d[i].append(j)
    if len(d[-1]) == 1 or len(d[1]) == 1:
        position = d[-1].pop() if len(d[-1]) == 1 else d[1].pop()
        return position
    else:
        return None


def comp_func(x, d):
    """ Determines the degree of geometric balancing for a particular point

    Calculates the level of geometric balancing for a particular point

    Args:
        x: The point of interest
        d: The position where there is a uniquely positive or negative element
           in an array.

    Returns:
        balance: The degree of balancing for x
    """

    dimension = len(x)
    avail = [i for i in range(dimension)]
    avail.remove(d)
    odd_ball = (x[d])
    denominator = 1
    for i in avail:
        denominator *= (x[i]-odd_ball)
    balance = abs((odd_ball**(dimension-1))/denominator)
    return balance


if __name__ == "__main__":
    # Parameters
    steps = 100
    n = 10000
    dimension = 12
    f = comp_func

    # Matplotlib stuff
    style.use('fivethirtyeight')
    plt.ion()
    fig, ax = plt.subplots(figsize=(20, 20))
    plot = ax.scatter([], [], s=10)
    ax.set_facecolor("white")
    fig.set_facecolor("white")
    plt.xlabel("$\\epsilon$", fontsize=26)
    h = plt.ylabel("$\\frac{hits}{N}$", fontsize=32, labelpad=20)
    h.set_rotation(0)
    box = dict(boxstyle='square, pad=0.4', facecolor='white',
               edgecolor='black', alpha=1)
    ax.text(.1, .1,
            "N="+str(n)+"\n$\\Delta\\epsilon=\\frac{1}{"+str(2*steps+1)+"}$",
            transform=ax.transAxes, fontsize=20, verticalalignment='top',
            bbox=box)
    ax.set_xlim((.5, 0))
    ax.set_ylim((0, 1))
    points = []
    for i in [(k+1)/(2*steps+1) for k in range(steps)][::-1]:
        point = np.array([i, check_epsilon(i, f, n, dimension)])
        array = plot.get_offsets()
        array = np.append(array, point)
        if array.ndim == 1:
            array = np.reshape(array, (-1, 2))
        plot.set_offsets(array)
        fig.canvas.draw()
        ax.set_xlim(array[:, 0].max() + .01, array[:, 0].min()-.01)
        ax.set_ylim(array[:, 1].min() - .01, array[:, 1].max()+.01)

    fig.savefig(datetime.datetime.now().strftime('%I%MW%m%d%Y')+".png",
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.ioff()
plt.show()
