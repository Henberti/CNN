import numpy as np
from functions import he_normal

class Dense:


    def __init__(self, n_neuron):
        self.n_neuron = n_neuron
        self.inputs = None
        self.weigth = None
        self.bias = None
        
    
    def forward(self, inputs):
        if self.inputs is None:
            self.weigth = he_normal((inputs.shape[1], self.n_neuron))
            self.bias = np.zeros((1,self.n_neuron))
        
        self.inputs = inputs
        
        return np.dot(input, self.weigth) + self.bias
    
    def backward(self, d_input):
        dweights = np.dot(self.inputs.T, d_input)
        dbiases = np.sum(d_input, axis=0, keepdims=True)
        return np.dot(d_input, self.weights.T)
    
    
        
        
            
            