import numpy as np
from core.model import LinearRegression1D_Bias
from core.optimizer import gd_step
from visuals.plot_components import plot_components

x = np.linspace(-5, 5, 300)
y = 1.5 * x + np.where(x > 0, 2.0, -1.0) + np.random.normal(0, 0.3, len(x))

model = LinearRegression1D_Bias()

for _ in range(1500):
    gd_step(model, x, y, lr=0.01)

plot_components(x, model)
