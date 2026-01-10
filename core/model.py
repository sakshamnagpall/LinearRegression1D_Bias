import numpy as np

class LinearRegression1D_Bias:
    """
    Linear Regression where bias is a function of input:
    y = w*x + alpha * tanh(beta*x + gamma)
    """

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal()
        self.alpha = rng.normal()
        self.beta = rng.normal()
        self.gamma = rng.normal()

    def bias(self, x: np.ndarray) -> np.ndarray:
        return self.alpha * np.tanh(self.beta * x + self.gamma)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.w * x + self.bias(x)

    def parameters(self):
        return {
            "w": self.w,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma
        }