import numpy as np


class MaxPool2x2:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        batch, channels, h, w = x.shape
        out_h = h // 2
        out_w = w // 2
        
        # Vectorized max pooling using reshape
        # Reshape to group 2x2 regions: (batch, channels, out_h, 2, out_w, 2)
        reshaped = x[:, :, :out_h*2, :out_w*2].reshape(batch, channels, out_h, 2, out_w, 2)
        # Move spatial dimensions together: (batch, channels, out_h, out_w, 2, 2)
        reshaped = reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(batch, channels, out_h, out_w, 4)
        
        out = np.max(reshaped, axis=4)
        argmax_flat = np.argmax(reshaped, axis=4)
        
        mask = np.zeros_like(x, dtype=bool)
        for n in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        flat_idx = argmax_flat[n, c, i, j]
                        row = flat_idx // 2
                        col = flat_idx % 2
                        mask[n, c, 2*i + row, 2*j + col] = True
        
        self.cache = mask
        return out

    def backward(self, grad_out):
        mask = self.cache
        batch, channels, out_h, out_w = grad_out.shape
        grad = np.zeros_like(mask, dtype=np.float32)
        
        for n in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        window_mask = mask[n, c, 2*i:2*i+2, 2*j:2*j+2]
                        grad[n, c, 2*i:2*i+2, 2*j:2*j+2][window_mask] = grad_out[n, c, i, j]
        
        return grad
