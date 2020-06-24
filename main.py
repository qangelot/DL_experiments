import numpy as np
from activation import * 
from costfn import * 


class NeuralNet:

    def __init__(self, size, costfn=QuadraticCost, activationfnHidden=SigmoidActivation, \
                activationfnLast=SigmoidActivation):
        '''
        size = list of integers specifying the number of nodes per layer. Includes input and output layers. 
        e.g.(100,50,10) is a three layers network with one input layer, one hidden layer and one output layer
        costfn = cost function for the network. Should be an instance of one of the cost function classes
        activationfnHidden = activation function for all of the hidden nodes. Should be an instance
        of one of the activation function classes
        activationfnLast = activation function for the nodes in the last (output) layer. Should be an 
        instance of one of the activation function classes
        '''
        self.weights = []
        for a, b in zip(size[:-1], size[1:]):
            self.weights.append(np.zeros((a,b)))
        self.biases = []
        for b in size[1:]:
            self.biases.append(np.zeros((1, b)))
        self.layers = len(size)
        self.costfn = costfn
        self.activationfnHidden = activationfnHidden
        self.activationfnLast = activationfnLast
        
    def initialize_variables(self):
        np.random.seed(1)
        i = 0
        for w in self.weights:
            self.weights[i] = (np.random.uniform(-1, 1, size=w.shape) / np.sqrt(w.shape[0]))
            i += 1
        i = 0
        for b in self.biases:
            self.biases[i] = np.random.uniform(-1, 1, size=b.shape)
            i += 1
            
    def initialize_variables_normalized(self):
        '''
        Normalized initialization
        Suggested for deep networks : more than 4-5 layers
        '''
        np.random.seed(1)
        i = 0
        for w in self.weights:
            self.weights[i] = (np.random.uniform(-1, 1, size=w.shape) * np.sqrt(6)\
                                    / np.sqrt(w.shape[0] + w.shape[1]))
            i += 1
        i = 0
        for b in self.biases:
            self.biases[i] = np.random.uniform(-1, 1, size=b.shape)
            i += 1
            
    def initialize_variables_alt(self):
        '''Appears to be effective for shallow networks (cross-entropy cost + ReLU hidden layers)'''
        np.random.seed(1)
        i = 0
        for w in self.weights:
            self.weights[i] = (np.random.normal(size=w.shape) / w.shape[1])
            i += 1
        i = 0
        for b in self.biases:
            self.biases[i] = np.random.normal(size=b.shape)
            i += 1

    def feedforward(self, data):
        '''
        data = batch of input data
        Assumes data is structured as an m x n numpy array, examples x features
        Returns neural network output for this batch of data
        '''
        
        z = data
        for w, b in zip(self.weights[0:-1], self.biases[0:-1]):
            z = self.activationfnHidden.fn(np.dot(z,w) + b)
        z = self.activationfnLast.fn(np.dot(z, self.weights[-1]) + self.biases[-1])
        return z
    
    def backprop(self, x, y, lmda):
        '''
        x = batch of input data
        y = correct output values for the batch
        lmda = regularization parameter
        z = weighted input to neurons in layer l
        a = activation of neurons in layer l
        
        backprop returns the current cost for the batch and two lists of matrices:
        dw = list of partial derivatives of cost w.r.t. weights per layer
        db = list of partial derivatives of cost w.r.t. biases per layer
        '''
        
        num_data_points = x.shape[0]
        z_vals = []
        a_vals = [x]
        
        ''' feedforward: storing all z and a values per layer '''
        
        activation = x
        for w, b in zip(self.weights[0:-1], self.biases[0:-1]):
            z = np.dot(activation,w) + b
            z_vals.append(z)
            activation = self.activationfnHidden.fn(z)
            a_vals.append(activation)
        z = np.dot(activation,self.weights[-1]) + self.biases[-1]
        z_vals.append(z)
        activation = self.activationfnLast.fn(z)
        a_vals.append(activation)

        cost = self.costfn.compute_cost(a_vals[-1], y, lmda, self.weights)
        cost_prime = self.costfn.cost_prime(a_vals[-1], y)
        
        ''' backprop: Errors per neuron calculated first starting at the last layer
            and working backwards through the networks, then partial derivatives 
            are calculated for each set of weights and biases
            deltas = neuron error per layer'''
        
        deltas = []
        
        if (self.costfn.cost_type=="QuadraticCost"):
            output_layer_delta = cost_prime * self.activationfnLast.prime(z_vals[-1])
            
        elif (self.costfn.cost_type=="CrossEntropyCost" or self.costfn.cost_type=="LogLikelihoodCost"):
            output_layer_delta = cost_prime
        
        else:
            print("Cost function not implemented")
            exit(1)
        
        deltas.insert(0, output_layer_delta)
        
        for i in range(1,self.layers-1):
            interim = np.dot(deltas[0], (np.transpose(self.weights[-i])))
            act_prime = self.activationfnHidden.prime(z_vals[-i-1])
            delta = np.multiply(interim, act_prime)
            deltas.insert(0, delta)
        
        db = []
        for i in range(len(deltas)):
            interim = np.sum(deltas[i], axis=0) / num_data_points
            db.append(np.reshape(interim, (1, interim.shape[0])))
        
        dw = []
        for i in range(0,self.layers-1):
            interim = np.dot(np.transpose(a_vals[i]), deltas[i])
            interim = interim / num_data_points
            dw.append(interim)
        
        return cost, db, dw
    
    def update_weights(self, dw, db, learning_rate, lmda, num_data_points):
        '''
        dw = list of partial derivatives of cost w.r.t. weights per layer
        db = list of partial derivatives of cost w.r.t. biases per layer
        learning_rate = learning rate hyperparamter, constrains size of parameter updates
        lmda = regularization paramter
        num_data_points = size of batch
        '''
        
        i = 0
        weight_mult = 1 - ((learning_rate * lmda) / num_data_points)
        for w, nw in zip(self.weights, dw):
            self.weights[i] = weight_mult * w - learning_rate * nw
            i += 1
        i = 0
        for b, nb in zip(self.biases, db):
            self.biases[i] = b - learning_rate * nb
            i += 1
            
    def predict(self, x):
        '''
        x = batch of input data, 2D matrix, examples x features
        Function returns a 2D matrix of output values in one-hot encoded form
        '''
        
        output = self.feedforward(x)
        if (output.shape[1]==1):
            '''If only one output, convert to 1 if value > 0.5'''
            low_indices = output <= 0.5
            high_indices = output > 0.5
            output[low_indices] = 0
            output[high_indices] = 1
        else:
            '''Otherwise set maximum valued output element to 1, the rest to 0'''    
            max_elem = output.max(axis=1)
            max_elem = np.reshape(max_elem, (max_elem.shape[0], 1))
            output = np.floor(output/ max_elem)
        return output
    
    def accuracy(self, x, y):
        '''
        x = batch of input data, 2D matrix, examples x features
        y = corresponding correct output values for the batch, 2D matrix, examples x outputs
        Function returns % of correct classified examples in the batch
        '''
        prediction = self.predict(x)
        num_data_points = x.shape[0]
        if (prediction.shape[1]==1):
            result = np.sum(prediction==y) / num_data_points
        else:
            result = np.sum(prediction.argmax(axis=1)==y.argmax(axis=1)) / num_data_points
        return result
    
    def SGD(self, x, y, valid_x, valid_y, learning_rate, epochs, reporting_rate, lmda=0, batch_size=10, verbose=False):
        '''
        x = training data, 2D matrix, examples x features
        y = corresponding correct output values for the training data, 2D matrix, examples x outputs
        valid_x = validation data, 2D matrix, examples x features
        valid_y = corresponding correct output values for the validation data, 2D matrix, examples x outputs
        learning_rate = learning rate hyperparamter, constrains size of parameter updates
        epochs = number of iterations through the entire training dataset
        reporting_rate = rate at which to report information about the model's performance. 
        If the reporting rate is 10, then information will be printed every 10 epochs
        lmda = regularization paramter
        batch_size = batch size per parameter update. If batch size = 25 and there are 1000 examples in the 
        training data then there will be 40 updates per epoch
        verbose: parameter controlling whether to print additional information. Useful for debugging
        Function returns two lists contraining the training and validation cost per parameter update
        '''

        training_cost = []
        valid_cost = []
        num_data_points = batch_size
        total_data_points = x.shape[0]
        output = self.feedforward(x)
        cost = self.costfn.compute_cost(output,y,lmda,self.weights)
        accuracy = self.accuracy(x, y)
        valid_accuracy = self.accuracy(valid_x, valid_y)
        print("Training cost at start of training is %.5f and accuracy is %3.2f%%" % (cost, accuracy * 100))
        print("Validation set accuracy is %3.2f%%" % (valid_accuracy * 100))
        if (verbose==True):
            print("First 10 output values are:")
            print(output[0:10,:])
        
        for i in range(epochs):
            data = np.hstack((x,y))
            input_dims = x.shape[1]
            output_dims = y.shape[1]
            np.random.shuffle(data)
            batches = []
            dw =[]
            db = []
            
            for k in range(0, (total_data_points - batch_size), batch_size):
                batch = data[k:(k+batch_size),:]
                batches.append(batch)
            num_batches = len(batches)
            for j in range(num_batches):
                batch_x = batches[j][:,:input_dims]
                batch_y = batches[j][:,input_dims:]
                if (batch_y.ndim == 1):
                    batch_y = np.reshape(batch_y, (batch_y.shape[0],1))
                cost, db, dw = self.backprop(batch_x, batch_y, lmda)
                self.update_weights(dw, db, learning_rate, lmda, num_data_points)

                '''Monitoring progress (or lack of...)'''
                training_cost.append(cost)
                valid_output = self.feedforward(valid_x)
                valid_c = self.costfn.compute_cost(valid_output,valid_y,lmda,self.weights)
                valid_cost.append(valid_c)
            
            if (i % reporting_rate == 0):
                output = self.feedforward(x)
                cost = self.costfn.compute_cost(output,y,lmda,self.weights)
                accuracy = self.accuracy(x, y)
                valid_accuracy = self.accuracy(valid_x, valid_y)
                print("Training cost in epoch %d is %.5f and accuracy is %3.2f%%" % (i, cost, accuracy * 100))
                print("Validation set accuracy is %3.2f%%" % (valid_accuracy * 100))
                if (verbose==True):
                    print("First 10 output values are:")
                    print(output[0:10,:])
                    print("Weight updates")
                    for i in range(len(self.weights)):
                        print(dw[i])
                    print("Bias updates")
                    for i in range(len(self.biases)):
                        print(db[i])
                    print("Weights")
                    for i in range(len(self.weights)):
                        print(self.weights[i])
                    print("Biases")
                    for i in range(len(self.biases)):
                        print(self.biases[i])
                
        '''Final results'''
        output = self.feedforward(x)
        cost = self.costfn.compute_cost(output,y,lmda,self.weights)
        prediction = self.predict(x)
        accuracy = self.accuracy(x, y)
        valid_accuracy = self.accuracy(valid_x, valid_y)
        print("Final test cost is %.5f" % cost)
        print("Accuracy on training data is %3.2f%%, and accuracy on validation data is %3.2f%%" %
                (accuracy * 100, valid_accuracy * 100))  
        
        return training_cost, valid_cost


net = NeuralNet((2,4,1), QuadraticCost, SigmoidActivation, 
                SigmoidActivation)
net.initialize_variables()

# loading data

import pickle
train_x = pickle.load(open("MNIST_train_x.pkl", 'rb'))
train_y = pickle.load(open("MNIST_train_y.pkl", 'rb'))
test_x = pickle.load(open("MNIST_test_x.pkl", 'rb'))
test_y = pickle.load(open("MNIST_test_y.pkl", 'rb'))


net2 = NeuralNet((784,100,10), LogLikelihoodCost, ReluActivation, SoftmaxActivation)
net2.initialize_variables_alt()

learning_rate = 0.0001
epochs = 101
reporting_rate = 20
lmda = 0.5
batch_size = 100

training_cost, valid_cost = net2.SGD(train_x, train_y, test_x, test_y, learning_rate, \
        epochs, reporting_rate, lmda, batch_size, verbose=False)

