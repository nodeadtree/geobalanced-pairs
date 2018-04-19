import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style


def check_epsilon(epsilon, compare_func, n):
    hits = 0
    alpha = [1,1,1,1]
    for i in range(n):
        x, y = np.random.dirichlet(alpha, size=2)
        z = np.array([y[0]-x[0], (y[0]+y[1])-(x[0]+x[1]), (y[0]+y[1]+y[2])-(x[0]+x[1]+x[2])])
        d = find_odd(z)
        if d is not None and 1/2-epsilon < compare_func(z, d) < 1/2+epsilon:
            hits = hits + 1
    return hits / n


def find_odd(x):
    c = x
    d = {-1: [], 1: []}
    for j, i in enumerate(np.sign(c)):
        if i in d:
            d[i].append(j)
    if len(d[-1]) != 0 and len(d[1]) != 0:
        return min([d[-1], d[1]], key=lambda x: len(x))[-1]
    else:
        return None


def comp_func(x, d):
    avail = [0, 1, 2]
    avail.remove(d)
    odd_ball = (x[d])
    reg_boye_1 = (x[avail[-1]]-odd_ball)
    reg_boye_2 = (x[avail[1]]-odd_ball)
    return odd_ball**2/(reg_boye_1*reg_boye_2)


if __name__ == "__main__":
    # Parameters
    steps = 1000
    n = 1000
    f = comp_func

    # Matplotlib stuff
    style.use('fivethirtyeight')
    plt.ion()
    fig, ax = plt.subplots(figsize=(20, 20))
    plot = ax.scatter([], [], s=10)
    ax.set_facecolor("white")
    fig.set_facecolor("white")
    plt.xlabel("$\\epsilon$", fontsize=26)
    h = plt.ylabel("$\\frac{hits}{n}$", fontsize=32, labelpad=20)
    h.set_rotation(0)
    box = dict(boxstyle='square, pad=0.4', facecolor='white', edgecolor='black', alpha=1)
    ax.text(.1,.1, "n="+str(n)+"\n$\\Delta\\epsilon=\\frac{1}{"+str(2*steps+1)+"}$", transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=box)
    ax.set_xlim((.5, 0))
    ax.set_ylim((0, 1))
    points = []
    for i in [(k+1)/(2*steps+1) for k in range(steps)][::-1]:
        point = np.array([i, check_epsilon(i, f, n)])
        array = plot.get_offsets()
        array = np.append(array, point)
        plot.set_offsets(array)
        fig.canvas.draw()
        ax.set_xlim(array[:, 0].max() + .01, array[:, 0].min()-.01)
        ax.set_ylim(array[:, 1].min() - .01, array[:, 1].max()+.01)

    fig.savefig(datetime.datetime.now().strftime('%I%MW%m%d%Y')+".png", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.ioff()
plt.show()
