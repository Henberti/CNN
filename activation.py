import numpy as np


class Relu:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, d_inputs):
        d_inputs[self.output <= 0] = 0
        return d_inputs