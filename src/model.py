from __future__ import annotations

import numpy as np


class LinearRegressionScratch:
    """
    Univariate linear regression (from scratch):
      y_hat = w*x + b
    Uses gradient descent minimizing MSE.
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.w = 0.0
        self.b = 0.0
        self.history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = X.reshape(-1, 1).astype(float)
        y = y.reshape(-1).astype(float)

        n = len(y)
        self.w, self.b = 0.0, 0.0
        self.history = []

        for _ in range(self.epochs):
            y_hat = (self.w * X[:, 0]) + self.b
            error = y_hat - y
            mse = np.mean(error ** 2)

            # gradients
            dw = (2.0 / n) * np.sum(error * X[:, 0])
            db = (2.0 / n) * np.sum(error)

            # update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            self.history.append(mse)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(-1, 1).astype(float)
        return (self.w * X[:, 0]) + self.b
