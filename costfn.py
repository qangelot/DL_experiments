import numpy as np

class QuadraticCost:
    ''' Cost = 1 / 2n sum_x || y - output ||^ 2
        Can be used with any activation function in the output layer, however sigmoid is preferred''' 
    
    cost_type = "QuadraticCost"
    
    @staticmethod
    def compute_cost(output, y, lmda, weights):
        
        '''Cost function cost'''
        num_data_points = output.shape[0]
        diff = y - output
        
        '''Regularization cost'''
        sum_weights = 0
        for w in weights:
            sum_weights += np.sum(np.multiply(w,w))
        regularization = (lmda * sum_weights) / (num_data_points * 2)
        
        return  np.sum(np.multiply(diff, diff)) / (2 * num_data_points) + regularization
    
    @staticmethod
    def cost_prime(output, y):
        '''Derivative of the quadratic cost function'''
        
        return output - y

    
class CrossEntropyCost:
    ''' Cost = -1 / n sum_x (y * ln(output) + (1 - y)*ln(1- output))
        Should be used with a sigmoid output layer'''
    
    cost_type = "CrossEntropyCost"
    
    @staticmethod
    def compute_cost(output, y, lmda, weights):
        '''Cost function cost'''
        num_data_points = output.shape[0]
        interim = y * np.log(output) + (1 - y) * np.log(1 - output)
        
        '''Regularization cost'''
        sum_weights = 0
        for w in weights:
            sum_weights += np.sum(np.multiply(w,w))
        regularization = (lmda * sum_weights) / (num_data_points * 2)
        
        return  (-1 / num_data_points) * np.sum(interim) + regularization
    
    @staticmethod
    def cost_prime(output, y):
        '''
        Derivative of the cross entropy cost function
        ASSUMES that only sigmoid activation units are used in the output layer
        Derivative is not correct for other output layer activation functions, such as the ReLU
        Any activation function in the hidden layer can be used
        '''
        return output - y


class LogLikelihoodCost:
    ''' 
    Cost = -1 / n ln output_c
    output_c is the output of the model for the correct answer, 
    this can be implemented by y * ln output_c since y will be 0 for the all but the correct answer
    Should be used with a softmax output layer
    '''
    
    cost_type = "LogLikelihoodCost"
    
    @staticmethod
    def compute_cost(output, y, lmda, weights):
        '''Cost function cost'''
        num_data_points = output.shape[0]
        interim = y * np.log(output)
        
        '''Regularization cost'''
        sum_weights = 0
        for w in weights:
            sum_weights += np.sum(np.multiply(w,w))
        regularization = (lmda * sum_weights) / (num_data_points * 2)
        
        return  (-1 / num_data_points) * np.sum(interim) + regularization
    
    @staticmethod
    def cost_prime(output, y):
        '''
        Derivative of the log likelihood cost function
        ASSUMES that only softmax activation units are used in the output layer
        Any activation function in the hidden layer can be used
        '''
        return output - y    


