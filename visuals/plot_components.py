import matplotlib.pyplot as plt
import numpy as np


def plot_components(x, model):
    linear = model.w * x
    bias = model.bias(x)
    total = model.predict(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, linear, label="Linear Component")
    plt.plot(x, bias, label="Bias Function")
    plt.plot(x, total, label="Final Prediction", linewidth=2)
    plt.legend()
    plt.title("Model Components")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
