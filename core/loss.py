import numpy as np

def mse_with_bias_penalty(y_true, y_pred, alpha, lambda_bias=0.01):
    """
    Mean Squared Error with L2 penalty on bias magnitude
    """
    mse = np.mean((y_true - y_pred) ** 2)
    bias_penalty = lambda_bias * (alpha ** 2)
    return mse + bias_penalty
