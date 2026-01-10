import numpy as np
from core.model import LinearRegression1D_Bias
from core.optimizer import gradient_descent_step

# sensor drift simulation
x = np.linspace(0, 100, 600)
drift = 0.05 * x * np.tanh(0.1 * (x - 50))
y = 1.2 * x + drift + np.random.normal(0, 1.0, size=len(x))

model = LinearRegression1D_Bias()

for _ in range(3000):
    gradient_descent_step(model, x, y, lr=0.005)

print("Sensor drift learned via bias function")

