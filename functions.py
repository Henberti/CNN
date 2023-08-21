import numpy as np
from scipy.signal import correlate2d, convolve2d

def he_normal(shape):
    n_in = shape[0]
    stddev = np.sqrt(2 / n_in)
    return np.random.normal(0, stddev, size=shape)

# can use mode full, valid, same
def fast_cross_correlation(input, kernel, stride=1, mode='valid'):
    cross_corr = correlate2d(input, kernel, mode)
    return cross_corr[::stride, ::stride]

# can use mode full, valid, same
def fast_convolution(input, kernel, stride=1 ,mode='full'):
    convolution = convolve2d(input, kernel, mode)
    return convolution[::stride, ::stride]


