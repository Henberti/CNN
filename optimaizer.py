import numpy as np

from fully_connected import Dense
from Convolution import Conv2



class Optimizer:
    
    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
        
    # Update parameters nueral or conv
    def update_params(self, layer):
        if isinstance(layer, Dense):
            layer.weigth += -self.current_learning_rate * layer.dweights
            layer.bias += -self.current_learning_rate * layer.dbiases
        elif isinstance(layer, Conv2):
            self.biases = self.biases - self.learning_rate * layer.d_bias
            self.kernels = self.kernels - self.learning_rate * layer.d_kernel
        
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
        