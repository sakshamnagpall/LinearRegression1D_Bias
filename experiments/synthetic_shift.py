import numpy as np
from core.model import LinearRegression1D_Bias
from core.optimizer import gradient_descent_step

# synthetic dataset with regime shift
x = np.linspace(-10, 10, 500)
y = 2.0 * x + np.where(x > 0, 3.0, -2.0) + np.random.normal(0, 0.5, size=len(x))

model = LinearRegression1D_Bias()

for epoch in range(2000):
    gradient_descent_step(model, x, y, lr=0.01)

print("Learned parameters:", model.parameters())