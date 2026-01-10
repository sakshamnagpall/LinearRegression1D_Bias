import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from core.optimizer import gradient_descent_step


def animate_training(model, x, y, steps=200, lr=0.01):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))

    def update(frame):
        gradient_descent_step(model, x, y, lr)
        line.set_data(x, model.predict(x))
        return line,

    ani = FuncAnimation(fig, update, frames=steps, blit=True)
    plt.show()
