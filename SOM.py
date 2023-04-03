import numpy as np

class SOM:
    def __init__(self, map_size, input_size, sigma, learning_rate, num_epochs):
        self.map_size = map_size
        self.input_size = input_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.random.rand(map_size[0], map_size[1], input_size)
        # print(self.weights)

    def get_distance(self, x, y):
        return np.linalg.norm(x - y)

    def get_bmu(self, x):
        distances = np.zeros((self.map_size[0], self.map_size[1]))
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                distances[i,j] = self.get_distance(x, self.weights[i,j])
        bmu = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu

    def get_neighborhood(self, bmu, sigma):
        neighborhood = np.zeros((self.map_size[0], self.map_size[1]))
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                neighborhood[i,j] = np.exp(-((i-bmu[0])**2 + (j-bmu[1])**2)/(2*sigma**2))
        return neighborhood

    def update_weights(self, x, bmu, neighborhood, learning_rate):
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                self.weights[i,j,:] += learning_rate * neighborhood[i,j] * (x - self.weights[i,j,:])

    def train(self, data):
        for epoch in range(self.num_epochs):
            for i, x in enumerate(data):
                bmu = self.get_bmu(x)
                neighborhood = self.get_neighborhood(bmu, self.sigma)
                self.update_weights(x, bmu, neighborhood, self.learning_rate)
            self.sigma *= 0.99
            self.learning_rate *= 0.99

    def predict(self, data):
        predictions = np.zeros((data.shape[0],))
        for i, x in enumerate(data):
            bmu = self.get_bmu(x)
            predictions[i] = bmu[0] * self.map_size[1] + bmu[1]
        return predictions

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# generate sample data
X, y = make_blobs(n_samples=200, centers=4, n_features=2, random_state=0)
# print(X)
# print(y)
# initialize SOM
map_size = (10, 10)
input_size = 2
sigma = 2.0
learning_rate = 0.1
num_epochs = 100
som = SOM(map_size, input_size, sigma, learning_rate, num_epochs)
# print(som.get_bmu(10))

# train SOM
som.train(X)

# predict cluster labels for sample data
predictions = som.predict(X)
print(predictions)
# plot results
# plt.scatter(X[:,0], X[:,1], c=predictions)
# plt.show()
