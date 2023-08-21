from Convolution import Conv2
from activation import Relu
from pooling import MaxPool
from flatten import Flatten
from fully_connected import Dense
from optimaizer import Optimizer
from loss import Activation_Softmax_Loss_CategoricalCrossentropy
import numpy as np

N_SAMPLES = 0

def modify_samples(n):
    global N_SAMPLES
    N_SAMPLES = n
    
    

class Cnn:
    
    def __init__(self, inputs, y_true):
        self.inputs = inputs / 255
        modify_samples(inputs.shape[0])
        self.conv = Conv2(2)
        self.relu = Relu()
        self.pool = MaxPool()
        self.flatten = Flatten()
        self.dense1 = Dense(64)
        self.activation1 = Relu()
        self.dense2 = Dense(10)
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
        self.y_true = y_true
        self.optimizer = Optimizer(decay=1e-2)
        
        
    
    def train(self,epoch=10):
        outputs=None
        print(self.y_true)
        for i in range(epoch):
            outputs = self.conv.forward(self.inputs)
            outputs = self.relu.forward(outputs)
            outputs = self.pool.forward(outputs)
            outputs = self.flatten.forward(outputs)
            outputs = self.dense1.forward(outputs)
            outputs = self.activation1.forward(outputs)
            outputs = self.dense2.forward(outputs)
            outputs = self.loss_activation.forward(outputs, self.y_true)
            print(f"loss={outputs}")
            # print(self.loss_activation.output)
            predictions = np.argmax(self.loss_activation.output, axis=1)
            print(predictions)
            print(np.mean(predictions == self.y_true))
            self.loss_activation.backward(self.loss_activation.output, self.y_true)
            outputs = self.loss_activation.dinputs
            outputs = self.dense2.backward(outputs)
            outputs = self.activation1.backward(outputs)
            outputs = self.dense1.backward(outputs)
            outputs = self.flatten.backward(outputs)
            outputs = self.pool.backward(outputs)
            outputs = self.relu.backward(outputs)
            outputs = self.conv.backward(outputs)
            self.optimizer.pre_update_params()
            self.optimizer.update_params(self.dense1)
            self.optimizer.update_params(self.dense2)
            self.optimizer.post_update_params()

        return outputs
    
    def activate(self,input,y_true):
        input = input / 255
        outputs = self.conv.forward(input)
        outputs = self.relu.forward(outputs)
        outputs = self.pool.forward(outputs)
        outputs = self.flatten.forward(outputs)
        outputs = self.dense1.forward(outputs)
        outputs = self.activation1.forward(outputs)
        outputs = self.dense2.forward(outputs)
        outputs = self.loss_activation.forward(outputs, y_true)
        print(f"loss={outputs}")
        # print(self.loss_activation.output)
        predictions = np.argmax(self.loss_activation.output, axis=1)
        print(predictions)
        print(np.mean(predictions == self.y_true))
        return predictions
        
        