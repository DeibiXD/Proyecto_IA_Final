import math
import numpy as np


class Conv2D:
    def __init__(self, in_channels=1, out_channels=2, kernel_size=3, lr=0.01):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # He initialization for ReLU
        limit = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.normal(0, limit, (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros(out_channels)
        self.lr = lr
        self.cache = None

    def forward(self, x):
        batch, channels, h, w = x.shape
        k = self.kernel_size
        out_h = h - k + 1
        out_w = w - k + 1
        out = np.zeros((batch, self.out_channels, out_h, out_w))
        
        for n in range(batch):
            for oc in range(self.out_channels):
                for ic in range(channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            out[n, oc, i, j] += np.sum(x[n, ic, i:i+k, j:j+k] * self.W[oc, ic])
                out[n, oc] += self.b[oc]
        
        self.cache = x
        return out

    def backward(self, grad_out):
        x = self.cache
        batch, channels, h, w = x.shape
        k = self.kernel_size
        _, _, out_h, out_w = grad_out.shape

        grad_x = np.zeros_like(x)
        grad_W = np.zeros_like(self.W)
        grad_b = np.sum(grad_out, axis=(0, 2, 3))

        for n in range(batch):
            for oc in range(self.out_channels):
                for ic in range(channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            region = x[n, ic, i:i + k, j:j + k]
                            grad_W[oc, ic] += grad_out[n, oc, i, j] * region
                            grad_x[n, ic, i:i + k, j:j + k] += grad_out[n, oc, i, j] * self.W[oc, ic]

        self.W -= self.lr * grad_W / batch
        self.b -= self.lr * grad_b / batch
        return grad_x
