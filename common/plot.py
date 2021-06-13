from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

x = []
y = []


def animate(i):
    x.append(np.random.randint(0, 100))
    plt.cla()
    plt.plot(x)


def plot_data():
    ani = FuncAnimation(plt.gcf(), animate)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_data()
