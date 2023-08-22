from Convolution import Conv2
from activation import Relu
from pooling import MaxPool
from flatten import Flatten
from fully_connected import Dense
from optimaizer import Optimizer, Optimizer_Adam
from loss import Activation_Softmax_Loss_CategoricalCrossentropy
import numpy as np
from scipy.ndimage import rotate

N_SAMPLES = 0

def modify_samples(n):
    global N_SAMPLES
    N_SAMPLES = n
    
    

class Cnn:
    
    def __init__(self, inputs, y_true):
        dataset = inputs / 255
        self.inputs = self.augment_images(dataset)
        modify_samples(inputs.shape[0])
        self.conv = Conv2(8)
        self.relu = Relu()
        self.pool = MaxPool()
        self.flatten = Flatten()
        self.dense1 = Dense(64,weight_regularizer_l2=5e-4,
                            bias_regularizer_l2=5e-4)
        self.activation1 = Relu()
        self.dense2 = Dense(10)
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
        self.y_true = y_true
        self.optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7)
        self.conv2 = Conv2(6, filter_shape=(2,2))
        self.relu2 = Relu()
        self.pool2 = MaxPool()
        
        
        
    
    def train(self,epoch=10):
        outputs=None
        print(self.y_true)
        for i in range(epoch):
            outputs = self.conv.forward(self.inputs)
            outputs = self.relu.forward(outputs)
            outputs = self.pool.forward(outputs)
            outputs = self.conv2.forward(outputs)
            outputs = self.relu2.forward(outputs)
            outputs = self.pool2.forward(outputs)
            outputs = self.flatten.forward(outputs)
            outputs = self.dense1.forward(outputs)
            outputs = self.activation1.forward(outputs)
            outputs = self.dense2.forward(outputs)
            outputs = self.loss_activation.forward(outputs, self.y_true)
            
            regularization_loss = \
                self.loss_activation.loss.regularization_loss(self.dense1) + \
                self.loss_activation.loss.regularization_loss(self.dense2)
            loss = outputs + regularization_loss
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
            outputs = self.pool2.backward(outputs)
            outputs = self.relu2.backward(outputs)
            outputs = self.conv2.backward(outputs)
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
        modify_samples(input.shape[0])
        outputs = self.conv.forward(input)
        outputs = self.relu.forward(outputs)
        outputs = self.pool.forward(outputs)
        outputs = self.conv2.forward(outputs)
        outputs = self.relu2.forward(outputs)
        outputs = self.pool2.forward(outputs)
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
    
    
    def augment_images(self,images):
        num_images = images.shape[0]
        num_to_augment = int(0.2 * num_images)
        
        indices_to_augment = np.random.choice(num_images, num_to_augment, replace=False)
        
        for idx in indices_to_augment:
            images[idx] = rotate(images[idx], angle=np.random.uniform(-15, 15), mode='nearest', reshape=False)
            
            brightness_factor = np.random.uniform(0.9, 1.1)
            images[idx] = np.clip(images[idx] * brightness_factor, 0, 255).astype(np.uint8)
        
        return images
        
        