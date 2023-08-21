import numpy as np
from scipy.ndimage import maximum_filter
import main

class MaxPool:
    
    def __init__(self, filter_size=2):
        self.input = None
        self.filter_size = filter_size
        
    
    def forward(self,input):
        self.input = input
       
        out_mat_shape = (((input.shape[2]-self.filter_size) // self.filter_size)+1)
        outputs = np.zeros((input.shape[0],out_mat_shape,out_mat_shape))
        print(input.shape)
        print(outputs.shape)
        
        for idx, image in enumerate(input):
            outputs[idx] = maximum_filter(image, size=2)[::2,::2]
        
        return outputs
        
        
        
        
    def backward(self, dL_dout):
        dL_dinput = np.zeros_like(self.input)

        pool_size = self.filter_size
        for n in range(main.N_SAMPLES):
            for i in range(0, self.input.shape[1], pool_size):
                for j in range(0, self.input.shape[2], pool_size):
                    
                    window = self.input[n, i:i+pool_size, j:j+pool_size]
             
                    max_pos = np.unravel_index(window.argmax(), window.shape)
                 
                    dL_dinput[n, i + max_pos[0], j + max_pos[1]] = dL_dout[n, i//pool_size, j//pool_size]
                    
        return dL_dinput


        