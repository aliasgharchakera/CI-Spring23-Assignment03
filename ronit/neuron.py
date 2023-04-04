import numpy as np



class Neuron:
    def __init__(self, weights, bias, xloc, yloc):
        self.weights = weights
        self.bias = bias
        self.xloc = xloc
        self.yloc = yloc

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        return self.sigmoid(total)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))  # sigmoid activation function
    
    def generateRandom(numNeurons, features,minData, maxData, height, width):
        neurons = []
        for i in range(height):
            for j in range(width):

                # We have to add the range of the data to the weights 
                # for k in range(features):
                neurons.append(Neuron(np.random.uniform(low=minData, high= maxData, size= features), np.random.rand(), i, j))
        return neurons