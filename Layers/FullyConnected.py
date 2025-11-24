import math
import numpy as np


class FullyConnected:
    def __init__(self, in_units, out_units, lr=0.01):
        # Xavier/Glorot initialization
        limit = math.sqrt(6.0 / (in_units + out_units))
        self.W = np.random.uniform(-limit, limit, (in_units, out_units))
        self.b = np.zeros(out_units)
        self.lr = lr
        self.cache = None

    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        x = self.cache
        grad_W = x.T @ grad_out
        grad_b = np.sum(grad_out, axis=0)
        grad_x = grad_out @ self.W.T
        batch = x.shape[0]
        self.W -= self.lr * grad_W / batch
        self.b -= self.lr * grad_b / batch
        return grad_x
