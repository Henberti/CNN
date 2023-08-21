import numpy as np
from functions import he_normal, fast_convolution, fast_cross_correlation
from concurrent.futures import ThreadPoolExecutor


class Conv2:
    
    def __init__(self,n_filters, filter_shape=(3,3), stride=1):
        self.n_filters = n_filters
        self.kernels = he_normal((n_filters, )+filter_shape)
        self.biases = np.zeros(n_filters)
        self.stride = stride
        self.inputs = None
        self.kernel_shape = filter_shape
        
    
    def forward(self, inputs):
        self.inputs = inputs
        self.num_channels = inputs.shape[1]
        
        self.out_mat_shape = tuple([(inputs.shape[x+2] - self.kernel_shape[x]) // self.stride + 1 for x in range(2)])
        outputs = np.zeros((inputs.shape[0], self.num_channels * self.n_filters, *self.out_mat_shape))
        
        batch_size = inputs.shape[0]
        split_size = batch_size // 4
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_results = [executor.submit(self._compute_subset, i, min(i + split_size, batch_size)) for i in range(0, batch_size, split_size)]
            
            for future in future_results:
                start, end, result = future.result()
                outputs[start:end] = result
                
        return outputs
    
    
    def _compute_subset(self, start_idx, end_idx):
        outputs = np.zeros((end_idx - start_idx, self.num_channels * self.n_filters, *self.out_mat_shape))
        
        for batch_idx in range(start_idx, end_idx):
            image = self.inputs[batch_idx]
            for channel_idx, filtered_image in enumerate(image):
                for kernel_idx, kernel in enumerate(self.kernels):
                    output_channel_idx = channel_idx * self.n_filters + kernel_idx
                    
                    outputs[batch_idx - start_idx, output_channel_idx] = fast_cross_correlation(filtered_image, kernel, self.stride) + self.biases[output_channel_idx]
        
        return start_idx, end_idx, outputs
    
    
    def backward(self,d_input):
        d_bias = np.zeros_like(self.biases)
        d_kernel = np.zeros_like(self.kernels)
        d_output = np.zeros_like(self.inputs)
        
        batch_size = d_input.shape[0]
        split_size = batch_size // 4
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            kernel_bias_future = executor.submit(self._compute_kernal_bias_gradiant, d_input)
            out_results = [executor.submit(self._compute_d_subset, i, min(i + split_size, batch_size), d_input) for i in range(0, batch_size, split_size)]
            
            d_bias, d_kernel = kernel_bias_future.result()

            for future in out_results:
                start, end, d_output_subset = future.result()
                self.inputs[start:end] = d_output_subset
                
        self.d_bias = d_bias
        self.d_kernel = d_kernel
                
                
        return d_output
    
    
    def _compute_kernal_bias_gradiant(self, d_input):
        d_bias = np.zeros_like(self.biases)
        d_kernel = np.zeros_like(self.kernels)
        
        batch_size = d_input.shape[0]
        
        for i, image in enumerate(d_input):
            for j, filter_img in enumerate(image):
                d_bias[j%self.n_filters] += np.sum(filter_img) / batch_size 
                d_kernel[j%self.n_filters] += fast_cross_correlation(self.inputs[i, j // self.n_filters],filter_img, self.stride) / batch_size
                
        return d_bias, d_kernel
        
        
    def _compute_d_subset(self, start_idx, end_idx, d_input):
        d_output = np.zeros((end_idx - start_idx, )+ self.inputs.shape[1:])
       
        for batch_idx in range(start_idx, end_idx):
            image = d_input[batch_idx]
            for channel_idx, filtered_image in enumerate(image):
                d_output[batch_idx - start_idx, channel_idx // self.n_filters] += fast_convolution(filtered_image, self.kernels[channel_idx % self.n_filters], self.stride) / self.n_filters
                
        return start_idx, end_idx, d_output
                
        
        
    
    
    
                
                
                
                
                
            
            
            
            
        
        
        
        
        
        
        
    
    
    
    
        
        
        
        
        
        
    
    
        
        


