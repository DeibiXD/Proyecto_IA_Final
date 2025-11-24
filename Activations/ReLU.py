import numpy as np


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, grad_out):
        grad = grad_out.copy()
        grad[self.cache <= 0] = 0
        return grad
