import numpy as np


def gd_step(model, x, y, lr=0.01):
    """
    Performs one gradient descent step
    """
    y_pred = model.predict(x)
    error = y_pred - y

    tanh_term = np.tanh(model.beta * x + model.gamma)
    sech2 = 1 - tanh_term ** 2

    dw = np.mean(error * x)
    dalpha = np.mean(error * tanh_term)
    dbeta = np.mean(error * model.alpha * x * sech2)
    dgamma = np.mean(error * model.alpha * sech2)

    model.w -= lr * dw
    model.alpha -= lr * dalpha
    model.beta -= lr * dbeta
    model.gamma -= lr * dgamma
