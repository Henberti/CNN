import numpy as np


class Flatten:
    def __init__(self):
        self.original_shape = None


    def forward(self, inputs):
        self.original_shape = inputs.shape
        batch_size = inputs.shape[0]
        flattened_output = inputs.reshape(batch_size, -1)

        return flattened_output

    def backward(self, d_input):
        d_input = d_input.reshape(self.original_shape)
        return d_input