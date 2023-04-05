import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

class SOM:
    def __init__(self, map_size, input_size, sigma, learning_rate, num_epochs, c=False, countries=None):
        self.map_size = map_size
        self.input_size = input_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.c = c
        self.weights = np.random.rand(map_size[0], map_size[1], input_size)
        if self.c:
            self.countries = countries
            self.countryMap = {}
            for country in countries:
                self.countryMap[country] = (0, 0)
        self.rgb = dict()
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
            self.bmu = list()
            for i, p in enumerate(data):
                bmu = self.get_bmu(p)
                self.bmu.append(bmu)
                if self.c:
                    self.countryMap[self.countries[i]] = bmu
                neighborhood = self.get_neighborhood(bmu, self.sigma)
                self.update_weights(p, bmu, neighborhood, self.learning_rate)
            self.sigma *= 0.9
            self.learning_rate *= 0.9

    def predict(self, data):
        predictions = np.zeros((data.shape[0],))
        for i in range(len(data)):
            bmu = self.get_bmu(data[i])
            predictions[i] = bmu[0] * self.map_size[1] + bmu[1]
        return predictions
    
    def color(self, var1, var2):
        figure = plt.figure(figsize=(self.map_size[0], self.map_size[1]))
        ax = figure.add_subplot(111, aspect='equal')
        ax.set_xlim((0, self.map_size[0]))
        ax.set_ylim((0, self.map_size[1]))
        plt.rcParams.update({'font.size': 5})
        for row in range(self.map_size[0]):
            for column in range(self.map_size[1]):
                weight = self.weights[row][column]
                rgb = np.zeros(3)
                for i in range(len(weight)):
                    if i % 3 == 0:
                        rgb[0] += weight[i] * var1
                    elif i % 3 == 1:
                        rgb[1] += weight[i] * var1
                    else:
                        rgb[2] += weight[i] * var1

                # Normalizing the rgb values
                rgb = rgb / rgb.sum()
                self.rgb[(row, column)] = rgb
                ax.add_patch(plt.Rectangle((row, column), 1, 1, facecolor=(
                rgb[0], rgb[1], rgb[2], 1), edgecolor='black'))
                # plt.show()
        gridData = []
        self.colourMatch = {}
        for i in range(len(self.bmu)):
            country = self.countries[i]
            bmu = self.bmu[i]
            self.colourMatch[country] = self.rgb[bmu[0], bmu[1]]
            centerx = bmu[0] + var2
            centery = bmu[1] + var2
            counter = 0
            while (centerx, centery) in gridData and counter < 4:
                centery = centery + var2
                counter += 1
            gridData.append((centerx, centery))
            plt.text(centerx, centery, country)
            
        worldMap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        fig, ax = plt.subplots(figsize=(10, 10))
        worldMap.plot(ax=ax, facecolor='lightgray', edgecolor='black')
        for i in self.colourMatch:
            color = self.colourMatch[i]
            # print(color, i)
            if i in worldMap["iso_a3"].tolist():
                worldMap[worldMap.iso_a3 == i].plot(color=color, ax=ax)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Self Organizing Map Grid View Visualization with Countries')
        plt.show()
        

# import numpy as np
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt

# # generate sample data
# X, y = make_blobs(n_samples=200, centers=4, n_features=3, random_state=0)
# # print(X)
# # print(y)
# # initialize SOM
# map_size = (10, 10)
# input_size = 3
# sigma = 2.0
# learning_rate = 0.1
# num_epochs = 100
# som = SOM(map_size, input_size, sigma, learning_rate, num_epochs)
# # print(som.get_bmu(10))

# # train SOM
# som.train(X)

# # predict cluster labels for sample data
# predictions = som.predict(X)
# print(predictions)
# plot results
# plt.scatter(X[:,0], X[:,1], c=predictions)
# plt.show()
