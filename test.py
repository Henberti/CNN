import numpy as np
from Convolution import Conv2
from functions import  fast_convolution, fast_cross_correlation
import time
from pooling import MaxPool
from scipy.ndimage import maximum_filter
import main

def conTest(_inputs, _n_filters, _filter_shape=(3,3), _stride=1):
    
    testConv = Conv2(_n_filters,_filter_shape,_stride)
    pool = MaxPool()
    main.modify_samples(_inputs.shape[0])
    kernels = testConv.kernels
    biases = testConv.biases
    fixed_inputs = _inputs
    start_class_time = time.time()
    class_out_forward = testConv.forward(fixed_inputs)
    end_class_time = time.time()
    print(f"class executed in {end_class_time - start_class_time} seconds")
    
    out_mat_shape = (((28-2) // 2)+1)
    # print(out_mat_shape)
    
    
    
    # start_test_time = time.time()
    # test_out_forward = []
    # for image in _inputs:
    #     for kernel in kernels:
    #         test_out_forward.append(fast_cross_correlation(image, kernel, _stride))
            
            
    # test_out_forward_reshape =  np.array(test_out_forward)
    
    
    # end_test_time = time.time()
    
    # print(f"test executed in {end_test_time - start_test_time} seconds")
    pool_out = []
    s_pool_time_test = time.time()
    for image in class_out_forward:
        pool_out.append(maximum_filter(image, size=2)[::2,::2])
    e_pool_time_test = time.time()
    print(f"pool test executed in {e_pool_time_test - s_pool_time_test} seconds")
    
    pool_out = np.array(pool_out)
    
    # print(pool_out.shape)
    
    # print(np.array_equal(test_out_forward_reshape,class_out_forward))
    
    # testConv.backward(class_out_forward)
    s_pool_time = time.time()
    p = pool.forward(class_out_forward)
    e_pool_time = time.time()
    
    
    print(f"ttt{p.shape}")
    
    print(np.array_equal(p,pool_out))

    bb = pool_out[2]
    
    pool_back = pool.backward(pool_out)
    aa = pool_back[2]
    print(aa.shape)
    print(pool_back[0])
    a = np.argwhere(pool_back[0] > 0)
    print(a)
    
    
    
    
    
    print(f"pool executed in {e_pool_time - s_pool_time} seconds")
    
    
   
        
    
    
    
   
    
    