"""
Logistic Regression
Author: Emmanuel Larralde
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Computes sigmoid of x"""
    return 1/(1 + np.exp(-np.clip(x, -10, 10)))

class LogisticRegression:
    """Trainable model for logistic regression"""
    def __init__(self, n: int) -> None:
        self.w = np.random.randn(n + 1)

    def train(self, train_x: np.ndarray, train_y: np.ndarray, optimizer: object, *args, **kwargs) -> dict:
        """Fits the model using train_x as input data and train_y as labels"""
        self.train_z = np.c_[train_x, np.ones(train_x.shape[0])]
        self.train_y = train_y
        opt = optimizer(self, **kwargs)
        training_history = opt.solve(**kwargs)
        return training_history

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Returns pi(z) with z an input with 1s appended"""
        return sigmoid(np.matmul(z, self.w).flatten())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the class of x"""
        z = np.c_[x, np.ones(x.shape[0])]
        return self.forward(z) > 0.5

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """Computes the accuracy of pi(x) given y"""
        y_hat = self.predict(x)
        return np.mean(np.abs(y_hat - y))

    def loss(self) -> np.ndarray:
        """Computes the binary cross entropy of the model"""
        y_hat = self.forward(self.train_z)
        bin_cross_entropy = -np.sum(
            self.train_y*np.log(y_hat) + 
            (1 - self.train_y)*np.log(1 - y_hat)
        )
        return bin_cross_entropy

    def gradient(self) -> np.ndarray:
        """Computes the current gradient of the binary cross entropy"""
        y_hat = self.forward(self.train_z)
        error = y_hat - self.train_y
        return np.matmul(self.train_z.T, error)
