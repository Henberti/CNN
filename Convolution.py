import numpy as np
from functions import he_normal, fast_convolution, fast_cross_correlation
from concurrent.futures import ThreadPoolExecutor
import main


class Conv2:
    
    def __init__(self,n_filters, filter_shape=(3,3), stride=1):
        self.n_filters = n_filters
        self.kernels = he_normal((n_filters, )+filter_shape)
        self.biases = np.zeros(n_filters)
        self.stride = stride
        self.inputs = None
        self.kernel_shape = filter_shape
        
    # fixed
    def forward(self, inputs):
        self.inputs = inputs
      
        out_mat_shape = tuple([(inputs.shape[x+1] - self.kernel_shape[x]) // self.stride + 1 for x in range(2)])
        outputs = np.zeros((inputs.shape[0] * self.n_filters, *out_mat_shape))
        idx = 0

        
        for image in self.inputs:
            for ker_idx, kernel in enumerate(self.kernels):
                outputs[idx] = fast_cross_correlation(image, kernel, self.stride) + self.biases[ker_idx]
                idx += 1
                
        return outputs
    
    # fixed
    def backward(self,d_input):
        d_bias = np.zeros_like(self.biases)
        d_kernel = np.zeros_like(self.kernels)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            kernel_bias_future = executor.submit(self._compute_kernal_bias_gradiant, d_input)
            out_results = executor.submit(self._compute_d_subset, d_input)
            
            d_bias, d_kernel = kernel_bias_future.result()
            d_output = out_results.result()
                
        self.d_bias = d_bias
        self.d_kernel = d_kernel
                
        return d_output
    
    # fixed
    def _compute_kernal_bias_gradiant(self, d_input):
        d_bias = np.zeros_like(self.biases)
        d_kernel = np.zeros_like(self.kernels)
        
        
        
        for j, image in enumerate(d_input):
            d_bias[j%self.n_filters] += np.sum(image) / main.N_SAMPLES 
            d_kernel[j%self.n_filters] += fast_cross_correlation(self.inputs[j // self.n_filters],image, self.stride) / main.N_SAMPLES
                
        return d_bias, d_kernel
        
    # fixed
    def _compute_d_subset(self, d_input):
        d_output = np.zeros_like(self.inputs, dtype=np.float32)
       
        for i, image in enumerate(d_input):
            d_output[i // self.n_filters] += fast_convolution(image, self.kernels[i % self.n_filters], self.stride) / self.n_filters
                
        return d_output
    
    
        
        
            
            
                
        
        
    
    
    
                
                
                
                
                
            
            
            
            
        
        
        
        
        
        
        
    
    
    
    
        
        
        
        
        
        
    
    
        
        


