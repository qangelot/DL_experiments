import numpy as np

class SigmoidActivation:
    
    @staticmethod
    def fn(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def prime(x):
        '''
        Derivative of the sigmoid function
        '''
        return np.multiply(SigmoidActivation.fn(x), (1 - SigmoidActivation.fn(x))) 
    
class ReluActivation:
    '''Should not be used in the output layer, only hidden layers'''
    
    @staticmethod
    def fn(x):
        y = np.copy(x)
        ltzero_indices = y<0
        y[ltzero_indices] = 0
        return y
    
    @staticmethod
    def prime(x):
        ''' Derivative of the RELU function'''
        y = np.copy(x)
        ltzero_indices = y<0
        other_indices = y>=0
        y[ltzero_indices] = 0
        y[other_indices] = 1
        return y

class LeakyReluActivation:
    '''Should not be used in the output layer, only hidden layers'''
    
    @staticmethod
    def fn(x):
        y = np.copy(x)
        ltzero_indices = y<0
        y[ltzero_indices] = y[ltzero_indices] * 0.1
        return y
    
    @staticmethod
    def prime(x):
        ''' Derivative of the LRELU function'''
        y = np.copy(x)
        ltzero_indices = y<0
        other_indices = y>=0
        y[ltzero_indices] = 0.1
        y[other_indices] = 1
        return y    
    
class SoftmaxActivation:
    
    @staticmethod
    def fn(x):
        '''Subtracting large constant from each of x values to prevent overflow'''
        y = np.copy(x)
        max_per_row = np.amax(y, axis=1)
        max_per_row = max_per_row.reshape((max_per_row.shape[0], 1))
        y = y - max_per_row
        '''Adding small constant to prevent underflow'''
        exp_y = np.exp(y) + 0.001
        exp_y_sum = np.sum(exp_y, axis=1)
        exp_y_sum = np.reshape(exp_y_sum,(exp_y_sum.shape[0],1))
        return exp_y / exp_y_sum
    
    @staticmethod
    def prime(x):
        '''
        Derivative of the softmax function
        '''
        sftmax = SoftmaxActivation.fn(x) 
        return np.multiply(sftmax, (1 - sftmax))


