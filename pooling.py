import numpy as np
from scipy.ndimage import maximum_filter
from concurrent.futures import ThreadPoolExecutor


class MaxPool:
    
    def __init__(self, filter_size=2):
        self.input = None
        self.filter_size = filter_size
        
    
    def forward(self,input):
        self.input = input
        batch_size, channel_size = input.shape[:2]
        split_size = batch_size // 4
        
        out_mat_shape = (((input.shape[3]-self.filter_size) // self.filter_size)+1)
        outputs = np.zeros((batch_size,channel_size,out_mat_shape,out_mat_shape))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_results = [executor.submit(self.max_pooling, i, min(i + split_size, batch_size),out_mat_shape ) for i in range(0, batch_size, split_size)]
            
            for future in future_results:
                start, end, result = future.result()
                outputs[start:end] = result
                
        return outputs
        
         
        
    def max_pooling(self, start_idx, end_idx, out_mat_shape):
        output = np.zeros((end_idx-start_idx, self.input.shape[1], out_mat_shape, out_mat_shape))
        for batch_idx in range(start_idx, end_idx):
            image = self.input[batch_idx]
            for channel_idx, filtered_image in enumerate(image):
                output[batch_idx - start_idx, channel_idx] = maximum_filter(filtered_image, size=self.filter_size)[::self.filter_size, ::self.filter_size]
        
        return start_idx, end_idx, output
        
        
        
    def backward(self, dout):
        dinput = np.zeros_like(self.input)
        batch_size, channel_size, _, _ = self.input.shape

        split_size = batch_size // 4

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_results = [executor.submit(self.maxpool_backward_slice, i, min(i + split_size, batch_size), dout) for i in range(0, batch_size, split_size)]
            
            for future in future_results:
                start, end, result = future.result()
                dinput[start:end] = result

        return dinput

    def maxpool_backward_slice(self, start_idx, end_idx, dout):
        dinput_slice = np.zeros((end_idx - start_idx, *self.input.shape[1:]))
        
        for batch_idx in range(start_idx, end_idx):
            for channel_idx in range(self.input.shape[1]):
                
                image = self.input[batch_idx, channel_idx]
                dout_slice = dout[batch_idx, channel_idx]

                for i in range(0, image.shape[0], self.filter_size):
                    for j in range(0, image.shape[1], self.filter_size):
                        
                        slice_ = image[i:i+self.filter_size, j:j+self.filter_size]
                        
                    
                        mask = (slice_ == np.max(slice_))
                        
                        
                        dinput_slice[batch_idx - start_idx, channel_idx, i:i+self.filter_size, j:j+self.filter_size] += mask * dout_slice[i // self.filter_size, j // self.filter_size]
        
        return start_idx, end_idx, dinput_slice

        